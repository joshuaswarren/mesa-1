/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include "asahi/lib/agx_nir_lower_vbo.h"
#include "asahi/lib/agx_linker.h"
#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_compiler.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include <math.h>
#include <string.h>
#include "util/half_float.h"
#include "util/format/u_format.h"
#include <stdio.h>
#include <stdlib.h>
#include "util/u_debug.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"VBO FAIL line%d: %s\n",__LINE__,#x); abort(); } } while (0)
static const nir_shader_compiler_options options = {0};
/* ------------------------------------------------------------------------ */
/* Exact reference conversions.                                              */
/*                                                                          */
/* These are the Vulkan fixed-point/float conversion rules evaluated exactly  */
/* (correct rounding, ties to even). Apple's own vertex fetch on a G17-class  */
/* GPU (M5 Pro, applegpu_g17s, Metal [[stage_in]]) returned exactly these     */
/* values for every code of every lane of every format below                  */
/* (.gpu-handoff/vs-fetch-20260926/host-oracle), which is why they stand in   */
/* for the native `normflt` / `unpack` results here.                         */
/* ------------------------------------------------------------------------ */
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* Correctly rounded n / d for |n| < 2^24, 0 < d < 2^24. g * d is exact in a
 * double for every float g in range, so the comparison is exact. */
static float
exact_div(int32_t n, uint32_t d)
{
   float f = (float)((double)n / (double)d);
   float best = f;
   double best_err = fabs((double)f * d - n);
   const float cands[2] = {nextafterf(f, -INFINITY), nextafterf(f, INFINITY)};
   for (unsigned i = 0; i < 2; ++i) {
      double err = fabs((double)cands[i] * d - n);
      if (err < best_err || (err == best_err && !(f2u(cands[i]) & 1) &&
                             (f2u(best) & 1))) {
         best = cands[i];
         best_err = err;
      }
   }
   return best;
}

static int32_t sext(uint32_t c, unsigned bits)
{
   return (int32_t)(c << (32 - bits)) >> (32 - bits);
}

static float ref_unorm(uint32_t c, unsigned bits)
{
   return exact_div((int32_t)c, (1u << bits) - 1);
}

static float ref_snorm(uint32_t c, unsigned bits)
{
   float f = exact_div(sext(c, bits), (1u << (bits - 1)) - 1);
   return f < -1.0f ? -1.0f : f;
}

/* Unsigned 11/10-bit float (5-bit exponent, 6/5-bit mantissa). */
static float ref_ufloat(uint32_t c, unsigned mbits)
{
   const uint32_t e = c >> mbits, m = c & ((1u << mbits) - 1);
   if (e == 31)
      return m ? NAN : INFINITY;
   if (e == 0)
      return ldexpf((float)m, -14 - (int)mbits);
   return ldexpf((float)((1u << mbits) | m), (int)e - 15 - (int)mbits);
}

/* Bitwise equality, except that any NaN matches any NaN (neither the
 * conversion instructions nor Vulkan preserve NaN payloads). */
static bool same_bits(uint32_t a, uint32_t b, bool is_float)
{
   if (is_float && isnan(u2f(a)) && isnan(u2f(b)))
      return true;
   return a == b;
}

static unsigned ulp_distance(uint32_t a, uint32_t b)
{
   int64_t x = (a >> 31) ? -(int64_t)(a & 0x7fffffff) : (int64_t)a;
   int64_t y = (b >> 31) ? -(int64_t)(b & 0x7fffffff) : (int64_t)b;
   return (unsigned)(x > y ? x - y : y - x);
}

/* ------------------------------------------------------------------------ */
/* Lowering helpers.                                                         */
/* ------------------------------------------------------------------------ */

/* A vertex shader reading `nr_attribs` attributes of `format` (4 lanes each,
 * as `type`), writing each to output i, lowered with the G17P path. */
static nir_shader *
lower_attribs(enum pipe_format format, unsigned stride, unsigned nr_attribs,
              nir_alu_type type, unsigned bit_size)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                 &agx_nir_options,
                                                 "G17P vertex fetch");
   struct agx_velem_key attrs[AGX_MAX_ATTRIBS] = {0};
   for (unsigned i = 0; i < nr_attribs; ++i) {
      nir_def *input = nir_load_input(&b, 4, bit_size, nir_imm_int(&b, 0),
                                      .base = i, .dest_type = type);
      nir_store_output(&b, input, nir_imm_int(&b, 0), .base = i,
                       .write_mask = 15, .src_type = type);
      attrs[i] = (struct agx_velem_key){.format = format, .stride = stride};
   }
   CHECK(agx_nir_lower_vbo_raw(b.shader, attrs,
                               (struct agx_robustness){.soft_fault = true},
                               false));
   return b.shader;
}

static void
fold(nir_shader *s)
{
   bool changed;
   do {
      changed = false;
      NIR_PASS(changed, s, nir_opt_constant_folding);
      NIR_PASS(changed, s, nir_opt_copy_prop);
      NIR_PASS(changed, s, nir_opt_dce);
   } while (changed);
}

#define MAX_LOADS AGX_MAX_ATTRIBS
struct loads {
   unsigned n;
   nir_intrinsic_instr *load[MAX_LOADS];
   nir_intrinsic_instr *output[MAX_LOADS];
   unsigned alu;      /* conversion ALU left in NIR (excluding vecs/moves) */
};

static struct loads
find_loads(nir_shader *s)
{
   struct loads l = {0};
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu) {
               nir_op op = nir_instr_as_alu(instr)->op;
               if (op != nir_op_vec2 && op != nir_op_vec3 && op != nir_op_vec4 &&
                   op != nir_op_mov && op != nir_op_imul && op != nir_op_iadd &&
                   op != nir_op_ult && op != nir_op_bcsel)
                  ++l.alu;
               continue;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_constant_agx) {
               CHECK(l.n < MAX_LOADS);
               l.load[l.n++] = intr;
            }
            if (intr->intrinsic == nir_intrinsic_store_output)
               l.output[nir_intrinsic_base(intr)] = intr;
         }
      }
   }
   return l;
}

static void
replace_load(nir_intrinsic_instr *load, const uint32_t *values)
{
   nir_builder b = nir_builder_at(nir_before_instr(&load->instr));
   nir_def *comps[4];
   for (unsigned c = 0; c < load->def.num_components; ++c)
      comps[c] = nir_imm_int(&b, values[c]);
   nir_def_rewrite_uses(&load->def,
                        nir_vec(&b, comps, load->def.num_components));
}

static void
output_bits(nir_intrinsic_instr *output, uint32_t out[4])
{
   CHECK(nir_src_is_const(output->src[0]));
   const nir_const_value *v = nir_src_as_const_value(output->src[0]);
   for (unsigned c = 0; c < 4; ++c)
      out[c] = nir_src_bit_size(output->src[0]) == 16 ? v[c].u16 : v[c].u32;
}

/* ------------------------------------------------------------------------ */
/* 1. Shape: which formats take a native conversion, and its NIR form.       */
/* ------------------------------------------------------------------------ */
enum shape { NATIVE_LOAD, HALVES, SOFTWARE };

static void
check_shape(enum pipe_format format, unsigned stride, enum shape shape,
            enum pipe_format load_format, unsigned load_count)
{
   const struct util_format_description *desc = util_format_description(format);
   const bool is_int = util_format_is_pure_integer(format);
   nir_shader *s = lower_attribs(format, stride, 1,
                                 is_int ? (util_format_is_pure_sint(format)
                                              ? nir_type_int32
                                              : nir_type_uint32)
                                        : nir_type_float32, 32);
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_dce);
   struct loads l = find_loads(s);
   CHECK(l.n == 1);
   nir_intrinsic_instr *load = l.load[0];
   const unsigned packed = (nir_intrinsic_base(load) &
                            AGX_NIR_MEMORY_PACKED_COUNT_MASK) >>
                           AGX_NIR_MEMORY_PACKED_COUNT_SHIFT;
   if (shape == NATIVE_LOAD) {
      /* One load in the format itself; no conversion ALU in NIR. */
      CHECK(nir_intrinsic_format(load) == load_format);
      CHECK(load->def.num_components == load_count);
      CHECK(load->def.bit_size == 32 && packed == 0);
      CHECK(l.alu == 0);
   } else {
      /* The software path's raw load, unchanged. */
      CHECK(packed != 0 || !desc->is_array);
      CHECK(nir_intrinsic_format(load) == load_format);
      CHECK(load->def.num_components == load_count);
      if (shape == HALVES) {
         /* Per lane: one unpack_32_2x16_split (free: a half-register view)
          * and at most one widening conversion. */
         CHECK(l.alu <= 2 * util_format_get_nr_components(format));
      }
   }
   ralloc_free(s);
}

/* ------------------------------------------------------------------------ */
/* 2. Lanes, swizzle and defaults, with the load's result substituted by the  */
/*    native conversion's (exact) values.                                    */
/* ------------------------------------------------------------------------ */
static void
check_lanes(enum pipe_format format, const uint32_t load[4],
            const float expected[4])
{
   nir_shader *s = lower_attribs(format, 16, 1, nir_type_float32, 32);
   struct loads l = find_loads(s);
   CHECK(l.n == 1 && l.output[0]);
   replace_load(l.load[0], load);
   fold(s);
   l = find_loads(s);
   uint32_t out[4];
   output_bits(l.output[0], out);
   for (unsigned c = 0; c < 4; ++c)
      CHECK(out[c] == f2u(expected[c]));
   ralloc_free(s);
}

/* ------------------------------------------------------------------------ */
/* 3. Exhaustive exactness of every conversion that stays in NIR: the 16-bit */
/*    widening conversions and the software fallback formats. The load is    */
/*    replaced by the raw word, sixteen attributes per shader, and the       */
/*    result folded after the same lower_fdiv algebraic the compiler runs.   */
/* ------------------------------------------------------------------------ */
typedef uint32_t (*ref_fn)(uint32_t code, unsigned lane);
struct sweep {
   const char *name;
   enum pipe_format format;
   unsigned lanes, code_bits[4];
   nir_alu_type type;
   ref_fn ref;
   bool is_float;
};

static unsigned
pack_word_bits(const struct sweep *sw, const uint32_t *codes, uint32_t words[2])
{
   words[0] = words[1] = 0;
   unsigned bit = 0;
   for (unsigned c = 0; c < sw->lanes; ++c) {
      const uint64_t v = (uint64_t)codes[c] << (bit % 32);
      words[bit / 32] |= (uint32_t)v;
      if ((bit % 32) + sw->code_bits[c] > 32)
         words[bit / 32 + 1] |= (uint32_t)(v >> 32);
      bit += sw->code_bits[c];
   }
   return DIV_ROUND_UP(bit, 32);
}

static void
run_sweep(const struct sweep *sw, unsigned *mismatch, unsigned *max_ulp)
{
   unsigned max_code = 0;
   for (unsigned c = 0; c < sw->lanes; ++c)
      max_code = MAX2(max_code, 1u << sw->code_bits[c]);
   *mismatch = *max_ulp = 0;
   /* Lane c of vertex i holds (i + c * step) mod 2^bits, so every lane sees
    * every code. */
   for (unsigned first = 0; first < max_code; first += AGX_MAX_ATTRIBS) {
      nir_shader *s = lower_attribs(sw->format, 16, AGX_MAX_ATTRIBS, sw->type, 32);
      bool progress;
      do {
         progress = false;
         NIR_PASS(progress, s, nir_opt_algebraic);
         NIR_PASS(progress, s, nir_opt_copy_prop);
      } while (progress);
      struct loads l = find_loads(s);
      CHECK(l.n == AGX_MAX_ATTRIBS);
      uint32_t codes[AGX_MAX_ATTRIBS][4];
      for (unsigned a = 0; a < AGX_MAX_ATTRIBS; ++a) {
         for (unsigned c = 0; c < sw->lanes; ++c)
            codes[a][c] = (first + a + c * (max_code / 4 + 1)) &
                          ((1u << sw->code_bits[c]) - 1);
         uint32_t words[2];
         pack_word_bits(sw, codes[a], words);
         replace_load(l.load[a], words);
      }
      fold(s);
      l = find_loads(s);
      for (unsigned a = 0; a < AGX_MAX_ATTRIBS; ++a) {
         uint32_t out[4];
         output_bits(l.output[a], out);
         for (unsigned c = 0; c < sw->lanes; ++c) {
            const uint32_t want = sw->ref(codes[a][c], c);
            if (!same_bits(out[c], want, sw->is_float)) {
               if (!*mismatch)
                  fprintf(stderr, "  %s lane %u code 0x%x: got 0x%08x want 0x%08x\n",
                          sw->name, c, codes[a][c], out[c], want);
               ++*mismatch;
               *max_ulp = MAX2(*max_ulp, ulp_distance(out[c], want));
            }
         }
      }
      ralloc_free(s);
   }
}

static uint32_t r_half(uint32_t c, unsigned l) { return f2u(_mesa_half_to_float(c)); }
static uint32_t r_s16(uint32_t c, unsigned l) { return (uint32_t)sext(c, 16); }
static uint32_t r_u16(uint32_t c, unsigned l) { return c; }
static uint32_t r_ss16(uint32_t c, unsigned l) { return f2u((float)sext(c, 16)); }
static uint32_t r_us16(uint32_t c, unsigned l) { return f2u((float)c); }
static uint32_t r_s8(uint32_t c, unsigned l) { return (uint32_t)sext(c, 8); }
static uint32_t r_u8(uint32_t c, unsigned l) { return c; }
static uint32_t r_ss8(uint32_t c, unsigned l) { return f2u((float)sext(c, 8)); }
static uint32_t r_us8(uint32_t c, unsigned l) { return f2u((float)c); }
static uint32_t r_s1010102(uint32_t c, unsigned l) { return (uint32_t)sext(c, l < 3 ? 10 : 2); }
static uint32_t r_u1010102(uint32_t c, unsigned l) { return c; }
static uint32_t r_sn1010102(uint32_t c, unsigned l) { return f2u(ref_snorm(c, l < 3 ? 10 : 2)); }
static uint32_t r_ss1010102(uint32_t c, unsigned l) { return f2u((float)sext(c, l < 3 ? 10 : 2)); }
static uint32_t r_us1010102(uint32_t c, unsigned l) { return f2u((float)c); }

/* ------------------------------------------------------------------------ */
/* 4. What the switch changes: the software conversion the driver used for   */
/*    the natively converted formats (nir_format_unpack_rgba after           */
/*    lower_fdiv), against the exact result the native instruction gives.    */
/* ------------------------------------------------------------------------ */
static void
software_vs_exact(const char *name, enum pipe_format format, unsigned lanes,
                  const unsigned *bits, bool snorm, unsigned *mismatch,
                  unsigned *max_ulp, unsigned *total)
{
   *mismatch = *max_ulp = *total = 0;
   unsigned max_code = 0;
   for (unsigned c = 0; c < lanes; ++c)
      max_code = MAX2(max_code, 1u << bits[c]);
   const unsigned batch = 64;
   for (unsigned first = 0; first < max_code; first += batch) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                    &agx_nir_options,
                                                    "software unpack");
      unsigned total_bits = 0;
      for (unsigned c = 0; c < lanes; ++c)
         total_bits += bits[c];
      const unsigned words = DIV_ROUND_UP(total_bits, 32);
      nir_def *raw[64];
      for (unsigned i = 0; i < batch; ++i) {
         raw[i] = nir_load_input(&b, words, 32, nir_imm_int(&b, 0), .base = i,
                                 .dest_type = nir_type_uint32);
         nir_def *rgba = nir_format_unpack_rgba(&b, raw[i], format);
         nir_store_output(&b, rgba, nir_imm_int(&b, 0), .base = i,
                          .write_mask = 15, .src_type = nir_type_float32);
      }
      bool progress;
      do {
         progress = false;
         NIR_PASS(progress, b.shader, nir_opt_algebraic);
         NIR_PASS(progress, b.shader, nir_opt_copy_prop);
      } while (progress);
      uint32_t codes[64][4];
      nir_intrinsic_instr *outputs[64] = {NULL};
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               const unsigned i = nir_intrinsic_base(intr);
               if (intr->intrinsic == nir_intrinsic_store_output)
                  outputs[i] = intr;
               if (intr->intrinsic != nir_intrinsic_load_input)
                  continue;
               struct sweep sw = {.lanes = lanes};
               for (unsigned c = 0; c < lanes; ++c)
                  sw.code_bits[c] = bits[c];
               for (unsigned c = 0; c < lanes; ++c)
                  codes[i][c] = (first + i + c * (max_code / 4 + 1)) &
                                ((1u << bits[c]) - 1);
               uint32_t w[2];
               pack_word_bits(&sw, codes[i], w);
               replace_load(intr, w);
            }
         }
      }
      fold(b.shader);
      for (unsigned i = 0; i < batch && first + i < max_code; ++i) {
         uint32_t out[4];
         output_bits(outputs[i], out);
         for (unsigned c = 0; c < lanes; ++c) {
            const uint32_t want = f2u(snorm ? ref_snorm(codes[i][c], bits[c])
                                            : ref_unorm(codes[i][c], bits[c]));
            ++*total;
            if (out[c] != want) {
               ++*mismatch;
               *max_ulp = MAX2(*max_ulp, ulp_distance(out[c], want));
            }
         }
      }
      ralloc_free(b.shader);
   }
   printf("software-vs-exact %-22s %6u / %6u lanes differ, max %u ULP\n", name,
          *mismatch, *total, *max_ulp);
}

static void check_prolog(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, &options,
                                                 "packed vertex prolog");
   struct agx_vs_prolog_key key = {.hw = true};
   key.attribs[0] = (struct agx_velem_key){.format = PIPE_FORMAT_R8G8B8A8_UNORM,
                                         .stride = 5};
   for (unsigned c = 0; c < 4; ++c) BITSET_SET(key.component_mask, c);
   agx_nir_vs_prolog_raw(&b, &key);
   unsigned loads = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic) continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_constant_agx) continue;
            ++loads;
            /* The prolog takes the same native conversion. */
            CHECK(nir_intrinsic_format(intr) == PIPE_FORMAT_R8_UNORM);
            CHECK(intr->def.num_components == 4);
            CHECK((nir_intrinsic_base(intr) & AGX_NIR_MEMORY_PACKED_COUNT_MASK) == 0);
         }
      }
   }
   CHECK(loads == 1);
   ralloc_free(b.shader);
}

/* Every native conversion must survive the G17P backend and packer: compile
 * a hardware vertex prolog fetching the format, all lanes. */
static void
check_compiles(enum pipe_format format, unsigned stride)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G17P native fetch prolog");
   struct agx_vs_prolog_key prolog = {
      .hw = true,
      .robustness = {.soft_fault = true},
   };
   prolog.attribs[0] = (struct agx_velem_key){.format = format, .stride = stride};
   for (unsigned c = 0; c < 4; ++c)
      BITSET_SET(prolog.component_mask, c);
   agx_nir_vs_prolog_raw(&b, &prolog);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_stop = true,
   };
   struct agx_shader_part part = {0};
   bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   if (!compiled)
      fprintf(stderr, "%s stride %u failed G17P prolog compilation\n",
              util_format_name(format), stride);
   CHECK(compiled);
   CHECK(part.binary != NULL && part.info.binary_size > 4);
   free(part.binary);
   ralloc_free(b.shader);
}

static void
check_instance_divisor_lowering(uint32_t divisor, uint32_t instance,
                                uint32_t base_instance,
                                uint32_t expected_index)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &options, "G17P instance divisor lowering");
   nir_def *input = nir_load_input(&b, 1, 32, nir_imm_int(&b, 0), .base = 0,
                                   .dest_type = nir_type_uint32);
   nir_store_output(&b, input, nir_imm_int(&b, 0), .base = 0, .write_mask = 1,
                    .src_type = nir_type_uint32);

   const struct agx_velem_key attrs[] = {{
      .divisor = divisor,
      .stride = 12,
      .format = PIPE_FORMAT_R32_UINT,
      .instanced = true,
   }};
   CHECK(agx_nir_lower_vbo_raw(
      b.shader, attrs, (struct agx_robustness){.soft_fault = true}, false));

   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_instance_id &&
                intr->intrinsic != nir_intrinsic_load_base_instance)
               continue;

            b.cursor = nir_before_instr(instr);
            nir_def *replacement =
               nir_imm_int(&b, intr->intrinsic == nir_intrinsic_load_instance_id
                                  ? instance
                                  : base_instance);
            nir_def_rewrite_uses(&intr->def, replacement);
         }
      }
   }

   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, b.shader, nir_opt_constant_folding);
      NIR_PASS(progress, b.shader, nir_opt_copy_prop);
      NIR_PASS(progress, b.shader, nir_opt_dce);
   } while (progress);

   unsigned loads = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_constant_agx)
               continue;

            ++loads;
            CHECK(nir_src_is_const(intr->src[1]));
            CHECK(nir_src_as_uint(intr->src[1]) == expected_index);
         }
      }
   }

   CHECK(loads == 1);
   ralloc_free(b.shader);
}

static void
check_instance_divisor_compiles(uint32_t divisor)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G17P instance divisor prolog");
   struct agx_vs_prolog_key prolog = {
      .hw = true,
      .robustness = {.soft_fault = true},
   };
   prolog.attribs[0] = (struct agx_velem_key){
      .divisor = divisor,
      .stride = 12,
      .format = PIPE_FORMAT_R32_UINT,
      .instanced = true,
   };
   BITSET_SET(prolog.component_mask, 0);
   agx_nir_vs_prolog_raw(&b, &prolog);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_stop = true,
   };
   struct agx_shader_part part = {0};
   bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   if (!compiled)
      fprintf(stderr, "divisor %u failed G17P prolog compilation\n", divisor);
   CHECK(compiled);
   CHECK(part.binary != NULL && part.info.binary_size > 4);

   bool found_instance_id = false;
   const uint8_t *binary = part.binary;
   for (unsigned i = 0; i + 4 <= part.info.binary_size; ++i) {
      if ((binary[i] & 0x7) == 0x4 && binary[i + 1] == 0xd8 &&
          binary[i + 2] == 0x10 && (binary[i + 3] & 0x1f) == 0x06) {
         found_instance_id = true;
         break;
      }
   }
   CHECK(found_instance_id);

   free(part.binary);
   ralloc_free(b.shader);
}
static void
check_uniform_sub_preamble(void)
{
   const bool enabled = debug_get_bool_option("AGX_G17P_VERTEX_UNIFORM_SUB", false);
   for (unsigned mode = 0; mode < 6; ++mode) {
      const bool g17p = mode != 1;
      mesa_shader_stage stage = mode == 2 ? MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX;
      unsigned bits = mode == 3 ? 16 : 32;
      unsigned comps = mode == 5 ? 2 : 1;
      nir_builder b = nir_builder_init_simple_shader(stage, &agx_nir_options,
                                                    "uniform negate cost");
      nir_def *uniform = nir_load_constant_agx(&b, comps, bits,
         nir_imm_int64(&b, 0x10000), nir_imm_int(&b, 0),
         .format = bits == 16 ? PIPE_FORMAT_R16_UINT : PIPE_FORMAT_R32_UINT);
      nir_def *neg = nir_ineg(&b, uniform);
      nir_def *varying = stage == MESA_SHADER_VERTEX
         ? nir_load_vertex_id(&b)
         : nir_f2u32(&b, nir_channel(&b, nir_load_frag_coord(&b), 0));
      varying = nir_u2uN(&b, varying, bits);
      nir_def *sum = mode == 4 ? neg : nir_iadd(&b, neg, varying);
      nir_store_output(&b, sum, nir_imm_int(&b, 0), .base = 0,
                       .write_mask = BITFIELD_MASK(comps));
      nir_divergence_analysis(b.shader);
      unsigned sizes[3] = {0, 0, 1};
      CHECK(agx_nir_opt_preamble(b.shader, sizes, g17p));
      unsigned main_neg = 0, pre_neg = 0;
      nir_function_impl *preamble = nir_shader_get_preamble(b.shader);
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_alu &&
                   nir_instr_as_alu(instr)->op == nir_op_ineg) {
                  if (impl == preamble) pre_neg++; else main_neg++;
               }
            }
         }
      }
      fprintf(stderr, "uniform-sub mode=%u enabled=%u main=%u pre=%u\n", mode, enabled, main_neg, pre_neg);
      if (enabled && (mode == 0 || mode == 5)) {
         CHECK(main_neg == 1 && pre_neg == 0);
      } else {
         CHECK(main_neg == 0 && pre_neg == 1);
      }
      ralloc_free(b.shader);
   }
}

static void
check_common_word_index(void)
{
   const bool enabled = debug_get_bool_option("AGX_G17P_VBO_COMMON_INDEX", false);
   for (unsigned mode = 0; mode < 4; ++mode) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
         &agx_nir_options, "mixed-width interleaved fetch index");
      struct agx_velem_key attrs[AGX_MAX_ATTRIBS] = {0};
      const enum pipe_format formats[] = {PIPE_FORMAT_R32G32B32_FLOAT,
         PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R16G16_SINT};
      for (unsigned i = 0; i < 3; ++i) {
         const unsigned nr = i == 0 ? 3 : i == 1 ? 4 : 2;
         nir_def *in = nir_load_input(&b, nr, 32, nir_imm_int(&b, 0),
                                      .base = i);
         nir_store_output(&b, in, nir_imm_int(&b, 0), .base = i,
                          .write_mask = BITFIELD_MASK(nr));
         attrs[i] = (struct agx_velem_key){.format = formats[i],
                                          .stride = mode == 2 ? 0 : 28};
      }
      struct agx_robustness rs = {.soft_fault = mode != 3,
         .level = mode == 1 ? AGX_ROBUSTNESS_GL : AGX_ROBUSTNESS_DISABLED};
      CHECK(agx_nir_lower_vbo_raw(b.shader, attrs, rs, mode == 2));
      for (unsigned i = 0; i < 3; ++i) { fold(b.shader); nir_opt_cse(b.shader); }
      struct loads l = find_loads(b.shader);
      CHECK(l.n == 3);
      for (unsigned i = 0; i < 3; ++i) {
         const unsigned field = nir_intrinsic_base(l.load[i]);
         CHECK(!!(field & AGX_NIR_MEMORY_INDEX_BYTES) == (enabled && mode == 0));
         if (enabled && mode == 0) {
            CHECK((field & 7) == 2);
            CHECK(l.load[i]->src[1].ssa == l.load[0]->src[1].ssa);
         }
      }
      if (enabled && mode == 0) {
         const uint32_t indices[] = {0, 1, 103, UINT32_MAX / 28,
            UINT32_MAX / 28 + 1, UINT32_MAX / 14 + 1, UINT32_MAX / 7};
         for (unsigned n = 0; n < ARRAY_SIZE(indices); ++n) {
            nir_shader *s = nir_shader_clone(NULL, b.shader);
            nir_foreach_function_impl(impl, s) {
               nir_foreach_block(block, impl) {
                  nir_foreach_instr_safe(instr, block) {
                     if (instr->type != nir_instr_type_intrinsic) continue;
                     nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                     if (intr->intrinsic != nir_intrinsic_load_vertex_id) continue;
                     nir_builder c = nir_builder_at(nir_before_instr(instr));
                     nir_def_rewrite_uses(&intr->def, nir_imm_int(&c, indices[n]));
                  }
               }
            }
            fold(s);
            struct loads actual = find_loads(s);
            CHECK(actual.n == 3);
            for (unsigned i = 0; i < actual.n; ++i) {
               CHECK(nir_src_is_const(actual.load[i]->src[1]));
               uint64_t word = nir_src_as_uint(actual.load[i]->src[1]);
               CHECK((word << 2) == (uint64_t)indices[n] * 28u);
            }
            ralloc_free(s);
         }
      }
      ralloc_free(b.shader);
   }
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   check_prolog();
   check_common_word_index();
   check_uniform_sub_preamble();

   /* 1. Which path each format takes. */
   const struct { enum pipe_format f; unsigned stride; enum shape shape;
                  enum pipe_format load; unsigned count; } shapes[] = {
      {PIPE_FORMAT_R8G8B8A8_UNORM, 28, NATIVE_LOAD, PIPE_FORMAT_R8_UNORM, 4},
      {PIPE_FORMAT_R8G8B8A8_UNORM, 5, NATIVE_LOAD, PIPE_FORMAT_R8_UNORM, 4},
      {PIPE_FORMAT_B8G8R8A8_UNORM, 16, NATIVE_LOAD, PIPE_FORMAT_R8_UNORM, 4},
      {PIPE_FORMAT_R8G8B8A8_SNORM, 36, NATIVE_LOAD, PIPE_FORMAT_R8_SNORM, 4},
      {PIPE_FORMAT_R8G8B8_SNORM, 36, NATIVE_LOAD, PIPE_FORMAT_R8_SNORM, 3},
      {PIPE_FORMAT_R8G8_UNORM, 4, NATIVE_LOAD, PIPE_FORMAT_R8_UNORM, 2},
      {PIPE_FORMAT_R8_SNORM, 4, NATIVE_LOAD, PIPE_FORMAT_R8_SNORM, 1},
      {PIPE_FORMAT_R16G16B16A16_UNORM, 8, NATIVE_LOAD, PIPE_FORMAT_R16_UNORM, 4},
      {PIPE_FORMAT_R16G16B16_SNORM, 6, NATIVE_LOAD, PIPE_FORMAT_R16_SNORM, 3},
      {PIPE_FORMAT_R16G16_SNORM, 4, NATIVE_LOAD, PIPE_FORMAT_R16_SNORM, 2},
      {PIPE_FORMAT_R16_UNORM, 4, NATIVE_LOAD, PIPE_FORMAT_R16_UNORM, 1},
      {PIPE_FORMAT_R10G10B10A2_UNORM, 4, NATIVE_LOAD, PIPE_FORMAT_R10G10B10A2_UNORM, 4},
      {PIPE_FORMAT_B10G10R10A2_UNORM, 4, NATIVE_LOAD, PIPE_FORMAT_R10G10B10A2_UNORM, 4},
      {PIPE_FORMAT_R11G11B10_FLOAT, 4, NATIVE_LOAD, PIPE_FORMAT_R11G11B10_FLOAT, 3},
      {PIPE_FORMAT_R16G16_SINT, 28, HALVES, PIPE_FORMAT_R16_UINT, 1},
      {PIPE_FORMAT_R16G16B16A16_UINT, 8, HALVES, PIPE_FORMAT_R16_UINT, 2},
      {PIPE_FORMAT_R16G16B16_FLOAT, 6, HALVES, PIPE_FORMAT_R16_UINT, 2},
      {PIPE_FORMAT_R16_FLOAT, 2, HALVES, PIPE_FORMAT_R16_UINT, 1},
      {PIPE_FORMAT_R16G16_SSCALED, 4, HALVES, PIPE_FORMAT_R16_UINT, 1},
      {PIPE_FORMAT_R16G16B16A16_USCALED, 8, HALVES, PIPE_FORMAT_R16_UINT, 2},
      /* No native conversion: unchanged software path. */
      {PIPE_FORMAT_R8G8B8A8_UINT, 4, SOFTWARE, PIPE_FORMAT_R8_UINT, 1},
      {PIPE_FORMAT_R8G8_SINT, 4, SOFTWARE, PIPE_FORMAT_R8_UINT, 1},
      {PIPE_FORMAT_R8G8B8A8_USCALED, 4, SOFTWARE, PIPE_FORMAT_R8_UINT, 1},
      {PIPE_FORMAT_R10G10B10A2_SNORM, 4, SOFTWARE, PIPE_FORMAT_R32_UINT, 1},
      {PIPE_FORMAT_R10G10B10A2_UINT, 4, SOFTWARE, PIPE_FORMAT_R32_UINT, 1},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(shapes); ++i)
      check_shape(shapes[i].f, shapes[i].stride, shapes[i].shape,
                  shapes[i].load, shapes[i].count);

   /* 2. Lanes, swizzles and default lanes after a native load. */
   check_lanes(PIPE_FORMAT_R8G8B8A8_UNORM,
               (uint32_t[]){f2u(32.f / 255), f2u(64.f / 255), f2u(128.f / 255), f2u(1)},
               (float[]){32.f / 255, 64.f / 255, 128.f / 255, 1});
   check_lanes(PIPE_FORMAT_B8G8R8A8_UNORM,
               (uint32_t[]){f2u(0.25f), f2u(0.5f), f2u(0.75f), f2u(1)},
               (float[]){0.75f, 0.5f, 0.25f, 1});
   check_lanes(PIPE_FORMAT_R8G8B8_SNORM,
               (uint32_t[]){f2u(-1), f2u(0), f2u(0.5f), 0},
               (float[]){-1, 0, 0.5f, 1});
   check_lanes(PIPE_FORMAT_R16G16_UNORM,
               (uint32_t[]){f2u(0.25f), f2u(1), 0, 0}, (float[]){0.25f, 1, 0, 1});
   check_lanes(PIPE_FORMAT_B10G10R10A2_UNORM,
               (uint32_t[]){f2u(0.125f), f2u(0.25f), f2u(0.5f), f2u(1)},
               (float[]){0.5f, 0.25f, 0.125f, 1});
   check_lanes(PIPE_FORMAT_R11G11B10_FLOAT,
               (uint32_t[]){f2u(2), f2u(0.5f), f2u(65024), 0},
               (float[]){2, 0.5f, 65024, 1});
   check_lanes(PIPE_FORMAT_R16G16_FLOAT, (uint32_t[]){0x40003c00, 0, 0, 0},
               (float[]){1, 2, 0, 1});

   /* 3. Exhaustive exactness of the conversions left in NIR. */
   const struct sweep sweeps[] = {
      {"R16G16B16A16_FLOAT", PIPE_FORMAT_R16G16B16A16_FLOAT, 4, {16, 16, 16, 16}, nir_type_float32, r_half, true},
      {"R16G16B16A16_SINT", PIPE_FORMAT_R16G16B16A16_SINT, 4, {16, 16, 16, 16}, nir_type_int32, r_s16, false},
      {"R16G16B16A16_UINT", PIPE_FORMAT_R16G16B16A16_UINT, 4, {16, 16, 16, 16}, nir_type_uint32, r_u16, false},
      {"R16G16B16A16_SSCALED", PIPE_FORMAT_R16G16B16A16_SSCALED, 4, {16, 16, 16, 16}, nir_type_float32, r_ss16, true},
      {"R16G16B16A16_USCALED", PIPE_FORMAT_R16G16B16A16_USCALED, 4, {16, 16, 16, 16}, nir_type_float32, r_us16, true},
      {"R16G16_SINT", PIPE_FORMAT_R16G16_SINT, 2, {16, 16}, nir_type_int32, r_s16, false},
      {"R16G16B16_FLOAT", PIPE_FORMAT_R16G16B16_FLOAT, 3, {16, 16, 16}, nir_type_float32, r_half, true},
      {"R8G8B8A8_SINT", PIPE_FORMAT_R8G8B8A8_SINT, 4, {8, 8, 8, 8}, nir_type_int32, r_s8, false},
      {"R8G8B8A8_UINT", PIPE_FORMAT_R8G8B8A8_UINT, 4, {8, 8, 8, 8}, nir_type_uint32, r_u8, false},
      {"R8G8B8A8_SSCALED", PIPE_FORMAT_R8G8B8A8_SSCALED, 4, {8, 8, 8, 8}, nir_type_float32, r_ss8, true},
      {"R8G8B8A8_USCALED", PIPE_FORMAT_R8G8B8A8_USCALED, 4, {8, 8, 8, 8}, nir_type_float32, r_us8, true},
      {"R10G10B10A2_SINT", PIPE_FORMAT_R10G10B10A2_SINT, 4, {10, 10, 10, 2}, nir_type_int32, r_s1010102, false},
      {"R10G10B10A2_UINT", PIPE_FORMAT_R10G10B10A2_UINT, 4, {10, 10, 10, 2}, nir_type_uint32, r_u1010102, false},
      {"R10G10B10A2_SSCALED", PIPE_FORMAT_R10G10B10A2_SSCALED, 4, {10, 10, 10, 2}, nir_type_float32, r_ss1010102, true},
      {"R10G10B10A2_USCALED", PIPE_FORMAT_R10G10B10A2_USCALED, 4, {10, 10, 10, 2}, nir_type_float32, r_us1010102, true},
      {"R10G10B10A2_SNORM", PIPE_FORMAT_R10G10B10A2_SNORM, 4, {10, 10, 10, 2}, nir_type_float32, r_sn1010102, true},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(sweeps); ++i) {
      unsigned mismatch, max_ulp;
      run_sweep(&sweeps[i], &mismatch, &max_ulp);
      printf("nir-vs-exact %-22s %u mismatching lanes (max %u ULP)\n",
             sweeps[i].name, mismatch, max_ulp);
      /* Software 10:10:10:2 SNORM keeps u2f * fl(1/511): at most 1 ULP. */
      if (sweeps[i].format == PIPE_FORMAT_R10G10B10A2_SNORM)
         CHECK(max_ulp <= 1);
      else
         CHECK(mismatch == 0);
   }

   /* 4. The conversions the native path replaces. */
   {
      unsigned m, u, t;
      software_vs_exact("R8G8B8A8_UNORM", PIPE_FORMAT_R8G8B8A8_UNORM, 4,
                        (unsigned[]){8, 8, 8, 8}, false, &m, &u, &t);
      CHECK(u <= 1);
      software_vs_exact("R8G8B8A8_SNORM", PIPE_FORMAT_R8G8B8A8_SNORM, 4,
                        (unsigned[]){8, 8, 8, 8}, true, &m, &u, &t);
      CHECK(u <= 1);
      software_vs_exact("R16G16_UNORM", PIPE_FORMAT_R16G16_UNORM, 2,
                        (unsigned[]){16, 16}, false, &m, &u, &t);
      CHECK(u <= 1);
      software_vs_exact("R16G16_SNORM", PIPE_FORMAT_R16G16_SNORM, 2,
                        (unsigned[]){16, 16}, true, &m, &u, &t);
      CHECK(u <= 1);
      software_vs_exact("R10G10B10A2_UNORM", PIPE_FORMAT_R10G10B10A2_UNORM, 4,
                        (unsigned[]){10, 10, 10, 2}, false, &m, &u, &t);
      CHECK(u <= 1);
   }

   /* The backend and packer accept every native conversion. */
   const struct { enum pipe_format f; unsigned stride; } compiles[] = {
      {PIPE_FORMAT_R8G8B8A8_UNORM, 28}, {PIPE_FORMAT_R8G8B8A8_UNORM, 5},
      {PIPE_FORMAT_B8G8R8A8_UNORM, 4}, {PIPE_FORMAT_R8G8B8A8_SNORM, 36},
      {PIPE_FORMAT_R8G8B8_SNORM, 36}, {PIPE_FORMAT_R8G8_UNORM, 2},
      {PIPE_FORMAT_R8_SNORM, 1}, {PIPE_FORMAT_R16G16B16A16_UNORM, 8},
      {PIPE_FORMAT_R16G16B16A16_SNORM, 8}, {PIPE_FORMAT_R16G16B16_UNORM, 6},
      {PIPE_FORMAT_R16G16_SNORM, 4}, {PIPE_FORMAT_R16_UNORM, 2},
      {PIPE_FORMAT_R10G10B10A2_UNORM, 4}, {PIPE_FORMAT_B10G10R10A2_UNORM, 4},
      {PIPE_FORMAT_R11G11B10_FLOAT, 4}, {PIPE_FORMAT_R16G16_SINT, 28},
      {PIPE_FORMAT_R16G16B16A16_SINT, 8}, {PIPE_FORMAT_R16G16B16_UINT, 6},
      {PIPE_FORMAT_R16_UINT, 2}, {PIPE_FORMAT_R16G16B16A16_FLOAT, 8},
      {PIPE_FORMAT_R16G16B16_FLOAT, 6}, {PIPE_FORMAT_R16_FLOAT, 2},
      {PIPE_FORMAT_R16G16_SSCALED, 4}, {PIPE_FORMAT_R16G16B16A16_USCALED, 8},
      {PIPE_FORMAT_R8G8B8A8_UINT, 4}, {PIPE_FORMAT_R10G10B10A2_SNORM, 4},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(compiles); ++i)
      check_compiles(compiles[i].f, compiles[i].stride);

   check_instance_divisor_lowering(0, UINT32_MAX, 5, 15);
   check_instance_divisor_lowering(1, 10, 5, 45);
   check_instance_divisor_lowering(2, 7, 5, 24);
   check_instance_divisor_lowering(3, 2, 5, 15);
   check_instance_divisor_lowering(3, 3, 5, 18);
   check_instance_divisor_lowering(3, 10, 5, 24);
   check_instance_divisor_lowering(UINT32_MAX, UINT32_MAX - 1, 0, 0);
   check_instance_divisor_lowering(UINT32_MAX, UINT32_MAX, 0, 3);
   check_instance_divisor_compiles(0);
   check_instance_divisor_compiles(1);
   check_instance_divisor_compiles(2);
   check_instance_divisor_compiles(3);
   check_instance_divisor_compiles(UINT32_MAX);
   glsl_type_singleton_decref();
   puts("G17P vertex fetch: native conversions, exactness sweeps and instance divisors PASS");
   return 0;
}
