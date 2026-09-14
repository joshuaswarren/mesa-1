/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G17P render passes with three to eight colour targets of arbitrary formats.
 *
 * Apple's builtin end-of-tile factory (metal-nt -arch applegpu_g17p, BTNF
 * root field 2 = 5) emits the same program for two to eight same-partition 2D
 * targets whatever their formats: the raw32 text for that target count in
 * agx3_g17p_quad_eot.h with each target's IMGWBLK selector byte replaced.
 * Sweeping internal formats 0..23 and components 1..4 shows the selector is
 * a function of the target's own format alone, and 42 mixed combinations of
 * two to four targets reproduce template-plus-selectors byte for byte. The
 * layouts below are the nine dEQP-VK renderpasses attachment.{3,4}.* cases,
 * followed by layouts of five to eight targets.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/format/u_format.h"
#include "agx3_g17p_quad_eot.h"
#include "agx_bg_eot_key.h"
#include "agx_compile.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "agx_tilebuffer.h"
#include "layout.h"
#include "nir_builder.h"

static unsigned checks;

#define CHECK(cond)                                                            \
   do {                                                                        \
      checks++;                                                                \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond);               \
         return 1;                                                             \
      }                                                                        \
   } while (0)

struct layout {
   const char *name;
   unsigned count;
   enum pipe_format formats[AGX_MAX_RENDER_TARGETS];
   uint8_t offsets[AGX_MAX_RENDER_TARGETS];
   uint8_t sample_size_B;
};

static const struct layout layouts[] = {
   {"3.296", 3,
    {PIPE_FORMAT_R16G16_SNORM, PIPE_FORMAT_R16_SNORM, PIPE_FORMAT_R16_FLOAT},
    {0, 4, 6}, 8},
   {"3.314", 3,
    {PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8_SINT},
    {0, 1, 5}, 8},
   {"3.336", 3,
    {PIPE_FORMAT_R16G16B16A16_SNORM, PIPE_FORMAT_R16_FLOAT,
     PIPE_FORMAT_R32G32B32A32_SINT},
    {16, 24, 0}, 32},
   {"3.397", 3,
    {PIPE_FORMAT_R16G16_SNORM, PIPE_FORMAT_R16G16_FLOAT,
     PIPE_FORMAT_R16G16B16A16_UINT},
    {0, 4, 8}, 16},
   {"4.418", 4,
    {PIPE_FORMAT_R8G8B8A8_SNORM, PIPE_FORMAT_R8G8B8A8_UNORM,
     PIPE_FORMAT_R32G32_UINT, PIPE_FORMAT_R16_UNORM},
    {10, 14, 0, 8}, 24},
   {"4.465", 4,
    {PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R16_UNORM,
     PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R8G8B8A8_UNORM},
    {0, 20, 4, 22}, 32},
   {"4.541", 4,
    {PIPE_FORMAT_R16G16B16A16_UINT, PIPE_FORMAT_R8_SINT,
     PIPE_FORMAT_R8G8_SNORM, PIPE_FORMAT_R8G8_SINT},
    {0, 8, 9, 11}, 16},
   {"4.568", 4,
    {PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R16G16B16A16_SNORM,
     PIPE_FORMAT_R8G8B8A8_SINT, PIPE_FORMAT_B8G8R8A8_UNORM},
    {0, 8, 16, 20}, 24},
   {"4.587", 4,
    {PIPE_FORMAT_R8G8B8A8_SRGB, PIPE_FORMAT_B10G10R10A2_UNORM,
     PIPE_FORMAT_B8G8R8A8_UNORM, PIPE_FORMAT_R16G16_UINT},
    {0, 4, 12, 8}, 16},
};

#define RGBA8 PIPE_FORMAT_R8G8B8A8_UNORM

/* Five to eight targets, all admitted by the mixed-format programs. The
 * homogeneous RGBA8 layouts match Apple's fragment witnesses. */
static const struct layout wide_layouts[] = {
   {"rgba8x5", 5, {RGBA8, RGBA8, RGBA8, RGBA8, RGBA8}, {0, 4, 8, 12, 16}, 24},
   {"mixed6", 6,
    {RGBA8, PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R8_UNORM,
     PIPE_FORMAT_R8G8B8A8_UINT, PIPE_FORMAT_R8G8_UNORM,
     PIPE_FORMAT_R32G32B32A32_FLOAT},
    {24, 16, 28, 29, 33, 0}, 40},
   {"rgba16x7", 7,
    {PIPE_FORMAT_R16G16B16A16_UNORM, PIPE_FORMAT_R16G16B16A16_UNORM,
     PIPE_FORMAT_R16G16B16A16_UNORM, PIPE_FORMAT_R16G16B16A16_UNORM,
     PIPE_FORMAT_R16G16B16A16_UNORM, PIPE_FORMAT_R16G16B16A16_UNORM,
     PIPE_FORMAT_R16G16B16A16_UNORM},
    {0, 8, 16, 24, 32, 40, 48}, 56},
   {"rgba8x8", 8, {RGBA8, RGBA8, RGBA8, RGBA8, RGBA8, RGBA8, RGBA8, RGBA8},
    {0, 4, 8, 12, 16, 20, 24, 28}, 32},
   {"mixed8", 8,
    {PIPE_FORMAT_R32G32B32A32_FLOAT, RGBA8, PIPE_FORMAT_R16G16B16A16_FLOAT,
     PIPE_FORMAT_R8G8B8A8_UINT, PIPE_FORMAT_R16G16_SNORM, PIPE_FORMAT_R8_UNORM,
     PIPE_FORMAT_R8G8_UNORM, PIPE_FORMAT_R32_FLOAT},
    {0, 32, 20, 36, 28, 40, 41, 16}, 48},
};

/* The store format the end-of-tile builder hands IMGWBLK. */
static enum pipe_format
eot_store_format(enum pipe_format format)
{
   return format == PIPE_FORMAT_R11G11B10_FLOAT ? PIPE_FORMAT_R32_UINT
          : util_format_is_srgb(format)        ? PIPE_FORMAT_R8_UNORM
                                         : ail_pixel_format[format].renderable;
}

static uint64_t
fnv1a(const uint8_t *data, size_t size)
{
   uint64_t hash = 0xcbf29ce484222325ull;
   for (size_t i = 0; i < size; ++i)
      hash = (hash ^ data[i]) * 0x100000001b3ull;
   return hash;
}

static bool
compile_eot(unsigned count, const enum pipe_format *formats,
            const uint8_t *offsets, const struct agx_shader_key *base_key,
            struct agx_shader_part *part)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "eot");
   nir_def *xy = nir_trim_vector(&b, nir_load_global_invocation_id(&b, 32), 2);
   for (unsigned rt = 0; rt < count; ++rt) {
      nir_image_store_block_agx(&b, nir_imm_intN_t(&b, rt, 16),
                                nir_imm_int(&b, offsets[rt]), xy,
                                .format = eot_store_format(formats[rt]),
                                .image_dim = GLSL_SAMPLER_DIM_2D,
                                .explicit_coord = true);
   }
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = *base_key;
   const bool ok = agx_compile_shader_nir(b.shader, &key, part);
   ralloc_free(b.shader);
   return ok;
}

/* Every admitted format's multi-target selector is the selector of that
 * format's own single-target program. */
static int
test_selectors_match_single_target_programs(void)
{
   unsigned admitted = 0;
   for (unsigned f = 1; f < PIPE_FORMAT_COUNT; ++f) {
      const uint8_t selector = agx_g17p_mrt_eot_selector(f);
      if (!selector)
         continue;

      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .no_preamble = true,
      };
      agx_g17p_configure_rt0_eot_key(f, &key);
      const enum pipe_format format = f;
      const uint8_t offset = 0;
      struct agx_shader_part part = {0};
      CHECK(compile_eot(1, &format, &offset, &key, &part));
      CHECK(part.info.binary_size == 68);
      CHECK(((uint8_t *)part.binary)[61] == selector);
      free(part.binary);
      admitted++;
   }

   /* The selector names the lane class Apple's own fragment stores the
    * format in (metal-nt render pipelines per MTLPixelFormat, matched to the
    * builtin background clears of each internal format). */
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R32G32_UINT) == 0x35);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R32G32_SINT) == 0x35);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R16G16B16A16_UNORM) == 0x95);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R16G16B16A16_SINT) == 0x15);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_B5G6R5_UNORM) == 0x15);
   /* ABGR4Unorm, A1BGR5Unorm and BGR5A1Unorm fragments store the same W16
    * float LSTCOL as RGBA16Float, RGB10A2Uint the same W16 integer LSTCOL as
    * RGBA16Uint, and RGBA8Snorm the same W8 LSTCOL as RGBA8Unorm. Channel
    * order is the PBE's, as for the BGRA forms already listed. */
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_A4B4G4R4_UNORM) == 0x15);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_A4R4G4B4_UNORM) == 0x15);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_B5G5R5A1_UNORM) == 0x15);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_B10G10R10A2_UINT) == 0x15);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_B8G8R8A8_SNORM) == 0x75);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R11G11B10_FLOAT) == 0);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R9G9B9E5_FLOAT) == 0);
   CHECK(admitted > 40);
   return 0;
}

static int
check_layout_classification(const struct layout *l)
{
   {
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
      memcpy(formats, l->formats, sizeof(l->formats));
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, l->count, 1, false);

      CHECK(tib.sample_size_B == l->sample_size_B);
      for (unsigned rt = 0; rt < l->count; ++rt)
         CHECK(agx_tilebuffer_offset_B(&tib, rt) == l->offsets[rt]);
      CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_MIXED);
      CHECK(agx_g17p_rt0_tile_layout_supported(&tib));

      struct agx_tilebuffer_layout bad = tib;
      bad.spilled[l->count - 1] = true;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      bad = tib;
      bad._offset_B[l->count - 1] = bad._offset_B[0];
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      bad = tib;
      bad._offset_B[l->count - 1] = tib.sample_size_B - 1;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      bad = tib;
      bad.logical_format[l->count - 1] = PIPE_FORMAT_R11G11B10_FLOAT;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      /* Multisample layouts take the mixed programs; other counts none. */
      bad = tib;
      bad.nr_samples = 2;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_MIXED);

      bad = tib;
      bad.nr_samples = 3;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      bad = tib;
      bad.layered = true;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);
   }
   return 0;
}

static int
test_layout_classification(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i) {
      if (check_layout_classification(&layouts[i]))
         return 1;
   }
   for (unsigned i = 0; i < ARRAY_SIZE(wide_layouts); ++i) {
      if (check_layout_classification(&wide_layouts[i]))
         return 1;
   }

   /* Targets that no longer fit one 64-byte sample spill, which G17P does not
    * support: eight RGBA32 targets. */
   enum pipe_format spill[AGX_MAX_RENDER_TARGETS];
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt)
      spill[rt] = PIPE_FORMAT_R32G32B32A32_UINT;
   struct agx_tilebuffer_layout spilled =
      agx_build_tilebuffer_layout(spill, AGX_MAX_RENDER_TARGETS, 1, false);
   CHECK(agx_g17p_classify_mrt(&spilled) == AGX_G17P_MRT_NONE);
   CHECK(!agx_g17p_rt0_tile_layout_supported(&spilled));

   /* Homogeneous layouts keep their own programs. */
   enum pipe_format rgba8[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM};
   enum pipe_format rgba32[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT};
   enum pipe_format r32[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32_FLOAT,
      PIPE_FORMAT_R32_FLOAT};
   for (unsigned count = 3; count <= 4; ++count) {
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(rgba8, count, 1, false);
      CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_RGBA8);
      tib = agx_build_tilebuffer_layout(rgba32, count, 1, false);
      CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_RGBA32);
   }
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(r32, 4, 1, false);
   CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_R32_QUAD);

   /* Two targets outside the two-target families use the same programs. */
   enum pipe_format pair[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R16_UNORM};
   tib = agx_build_tilebuffer_layout(pair, 2, 1, false);
   CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_MIXED);
   CHECK(agx_g17p_rt0_tile_layout_supported(&tib));

   /* A full sample selects the 16x16 tile, as for four RGBA32 targets. */
   enum pipe_format wide[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_R32G32B32A32_SINT, PIPE_FORMAT_R32G32B32A32_FLOAT};
   tib = agx_build_tilebuffer_layout(wide, 4, 1, false);
   CHECK(tib.tile_size == 16 * 16);
   CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_MIXED);
   CHECK(agx_g17p_rt0_tile_layout_supported(&tib));
   return 0;
}

/* Expected multi-target EOT: the native raw32 text with each target's
 * IMGWBLK selector at the byte metal-nt changes for it. */
static const uint8_t *
expected_eot(unsigned count, const uint8_t *selectors, uint8_t *out)
{
   static const unsigned selector_offsets[][AGX_MAX_RENDER_TARGETS] = {
      [2] = {101, 113},
      [3] = {133, 145, 157},
      [4] = {165, 177, 189, 201},
      [5] = {197, 209, 221, 233, 245},
      [6] = {229, 241, 253, 265, 277, 289},
      [7] = {69, 273, 285, 297, 309, 321, 333},
      [8] = {101, 113, 317, 329, 341, 353, 365, 377},
   };
   static const uint8_t *const texts[] = {
      [2] = agx3_g17p_dual_raw32_eot,  [3] = agx3_g17p_triple_raw32_eot,
      [4] = agx3_g17p_quad_eot,        [5] = agx3_g17p_penta_raw32_eot,
      [6] = agx3_g17p_hexa_raw32_eot,  [7] = agx3_g17p_hepta_raw32_eot,
      [8] = agx3_g17p_octa_raw32_eot,
   };
   memcpy(out, texts[count], 32 + 44 * count);
   for (unsigned rt = 0; rt < count; ++rt)
      out[selector_offsets[count][rt]] = selectors[rt];
   return out;
}

/* The multisample program for texture type 4 attachments changes three
 * bytes per target: the wire width byte of its PACK_IMGWIR gains 2, and its
 * IMGWBLK clears bit 7 of byte 8 and takes dimension class 2 in the selector's
 * low nibble. */
static const uint8_t *
expected_ms_eot(unsigned count, const uint8_t *selectors, uint8_t *out)
{
   static const unsigned wire_offsets[][AGX_MAX_RENDER_TARGETS] = {
      [2] = {43, 63},
      [3] = {43, 63, 95},
      [4] = {43, 63, 95, 127},
      [5] = {43, 63, 95, 127, 159},
      [6] = {43, 63, 95, 127, 159, 191},
      [7] = {43, 75, 107, 139, 171, 203, 235},
      [8] = {43, 63, 119, 151, 183, 215, 247, 279},
   };
   static const unsigned selector_offsets[][AGX_MAX_RENDER_TARGETS] = {
      [2] = {101, 113},
      [3] = {133, 145, 157},
      [4] = {165, 177, 189, 201},
      [5] = {197, 209, 221, 233, 245},
      [6] = {229, 241, 253, 265, 277, 289},
      [7] = {69, 273, 285, 297, 309, 321, 333},
      [8] = {101, 113, 317, 329, 341, 353, 365, 377},
   };
   expected_eot(count, selectors, out);
   for (unsigned rt = 0; rt < count; ++rt) {
      out[wire_offsets[count][rt]] |= 0x02;
      out[selector_offsets[count][rt] - 1] &= 0x7f;
      out[selector_offsets[count][rt]] = (selectors[rt] & 0xf0) | 0x02;
   }
   return out;
}

static unsigned
decode_hex(const char *hex, uint8_t *out, unsigned capacity)
{
   unsigned n = 0;
   for (; hex[0] && hex[1] && n < capacity; hex += 2)
      sscanf(hex, "%2hhx", &out[n++]);
   return n;
}

/* metal-nt -arch applegpu_g17p builtin end-of-tile BTNF: two, seven and eight
 * RGBA8 (internal format 2) targets in partition zero, texture type 4.
 * .tmp/cts-renderpasses-current-dd52/mixed/native/mseot_same_n{2,7,8}_t4. */
static const struct {
   unsigned count;
   const char *hex;
} ms_eot_witnesses[] = {
   {2,
      "1c9c100624a814060c9d102624a910269f1054020304140050260a009f20540003001000"
      "50260a009705540600080000d04010000b8221041b8021049705540e00100000d0441000"
      "4b032f004200000400005b012f0042000004000057105400000000802872000057205408"
      "00100090287200000e000000"},
   {7,
      "1c9c100624a814060c9d102624a910269f1054020304140050260a009f20540003001000"
      "50260a009705540600080000d04010000b8221041b802104571054000000008028720000"
      "9705540e00100000d04410004b8327004200000400005b81270042000004000097055416"
      "00180000d05010008b8327004200000800009b8127004200000800009705541e00200000"
      "d0541000cb8327004200000c0000db8127004200000c00009705542600280000d0601000"
      "0b8367004200001000001b8167004200001000009705542e00300000d06010004b836700"
      "4200001000005b8167004200001000009705543600380000d06410008b036f0042000014"
      "00009b016f0042000014000057205408001000902872000057405410002000a028720000"
      "57805418003000a02872000057005520004000b02872000057005428005000d028720000"
      "57005630006000e0287200000e000000"},
   {8,
      "1c9c100624a814060c9d102624a910269f1054020304140050260a009f20540003001000"
      "50260a009705540600080000d04010000b8221041b8021049705540e00100000d0441000"
      "4b8327004200000400005b81270042000004000057105400000000802872000057205408"
      "00100080287200009705541600180000d05010008b8327004200000800009b8127004200"
      "000800009705541e00200000d0541000cb8327004200000c0000db8127004200000c0000"
      "9705542600280000d06010000b8367004200001000001b8167004200001000009705542e"
      "00300000d06410004b8367004200001400005b8167004200001400009705543600380000"
      "d06410008b8367004200001400009b8167004200001400009705543e00400000d0441000"
      "cb036f00420000040000db016f0042000004000057405410002000a02872000057805418"
      "003000b02872000057005520004000b02872000057005628005000d02872000057005430"
      "006000e0287200005720543800700090287200000e000000"},
};

static int
test_multisample_eot_witnesses(void)
{
   const uint8_t rgba8[AGX_MAX_RENDER_TARGETS] = {
      0x75, 0x75, 0x75, 0x75, 0x75, 0x75, 0x75, 0x75,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(ms_eot_witnesses); ++i) {
      const unsigned count = ms_eot_witnesses[i].count;
      uint8_t native[sizeof(agx3_g17p_octa_raw32_eot)];
      uint8_t expected[sizeof(agx3_g17p_octa_raw32_eot)];
      CHECK(decode_hex(ms_eot_witnesses[i].hex, native, sizeof(native)) ==
            32 + 44 * count);
      CHECK(!memcmp(expected_ms_eot(count, rgba8, expected), native,
                    32 + 44 * count));
   }
   return 0;
}

static int
check_eot_subset(unsigned count, const enum pipe_format *formats,
                 const uint8_t *offsets)
{
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
      .g17p_mrt_eot_count = count,
   };
   agx_g17p_configure_rt0_eot_key(formats[0], &key);
   for (unsigned rt = 0; rt < count; ++rt)
      key.g17p_mrt_eot_selectors[rt] = agx_g17p_mrt_eot_selector(formats[rt]);

   struct agx_shader_part part = {0};
   CHECK(compile_eot(count, formats, offsets, &key, &part));
   uint8_t expected[sizeof(agx3_g17p_octa_raw32_eot)];
   CHECK(part.info.binary_size == 32 + 44 * count);
   CHECK(part.info.nr_gprs == 8);
   CHECK(part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 * count);
   CHECK(!memcmp(part.binary,
                 expected_eot(count, key.g17p_mrt_eot_selectors, expected),
                 part.info.binary_size));
   free(part.binary);

   struct agx_shader_key ms_key = key;
   ms_key.g17p_rt0_eot_multisample = true;
   part = (struct agx_shader_part){0};
   CHECK(compile_eot(count, formats, offsets, &ms_key, &part));
   CHECK(part.info.binary_size == 32 + 44 * count);
   CHECK(!memcmp(part.binary,
                 expected_ms_eot(count, key.g17p_mrt_eot_selectors, expected),
                 part.info.binary_size));
   free(part.binary);

   /* A store whose format cannot carry its target's selector is refused
    * rather than published with a selector from another format. */
   for (unsigned rt = 0; rt < count; ++rt) {
      struct agx_shader_key wrong = key;
      wrong.g17p_mrt_eot_selectors[rt] =
         key.g17p_mrt_eot_selectors[rt] == 0x55 ? 0x95 : 0x55;
      part = (struct agx_shader_part){0};
      CHECK(!compile_eot(count, formats, offsets, &wrong, &part));
   }

   struct agx_shader_key short_key = key;
   short_key.g17p_mrt_eot_count = count - 1;
   part = (struct agx_shader_part){0};
   CHECK(!compile_eot(count, formats, offsets, &short_key, &part));
   return 0;
}

static int
check_eot_programs(const struct layout *l, bool every_subset)
{
   {

      /* One stored target keeps its format's single-target program at its
       * own offset. */
      for (unsigned rt = 0; rt < l->count; ++rt) {
         struct agx_shader_key key = {
            .dev.encoding = AGX_ISA_ENCODING_AGX3B,
            .no_preamble = true,
         };
         agx_g17p_configure_rt0_eot_key(l->formats[rt], &key);
         struct agx_shader_part part = {0};
         CHECK(compile_eot(1, &l->formats[rt], &l->offsets[rt], &key, &part));
         CHECK(part.info.binary_size == 68);
         CHECK(((uint8_t *)part.binary)[61] ==
               agx_g17p_mrt_eot_selector(l->formats[rt]));
         free(part.binary);
      }

      for (unsigned mask = 1; mask < BITFIELD_MASK(l->count); ++mask) {
         if (util_bitcount(mask) < 2 ||
             (!every_subset && util_bitcount(mask) != l->count - 1))
            continue;
         enum pipe_format formats[AGX_MAX_RENDER_TARGETS];
         uint8_t offsets[AGX_MAX_RENDER_TARGETS];
         unsigned count = 0;
         u_foreach_bit(rt, mask) {
            formats[count] = l->formats[rt];
            offsets[count++] = l->offsets[rt];
         }
         if (check_eot_subset(count, formats, offsets))
            return 1;
      }
      if (check_eot_subset(l->count, l->formats, l->offsets))
         return 1;
   }
   return 0;
}

static int
test_eot_programs(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i) {
      if (check_eot_programs(&layouts[i], true))
         return 1;
   }
   /* Wider layouts: every target count through eight, without each subset. */
   for (unsigned i = 0; i < ARRAY_SIZE(wide_layouts); ++i) {
      if (check_eot_programs(&wide_layouts[i], false))
         return 1;
   }

   /* Homogeneous RGBA8 targets reproduce the separately captured programs. */
   const uint8_t rgba8[] = {0x75, 0x75, 0x75, 0x75};
   uint8_t expected[sizeof(agx3_g17p_octa_raw32_eot)];
   CHECK(!memcmp(expected_eot(3, rgba8, expected), agx3_g17p_rgba8_3_eot,
                 sizeof(agx3_g17p_rgba8_3_eot)));
   CHECK(!memcmp(expected_eot(4, rgba8, expected), agx3_g17p_rgba8_4_eot,
                 sizeof(agx3_g17p_rgba8_4_eot)));
   return 0;
}

static bool
lower_tex_handle_to_u0(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_texture_handle_agx)
      return false;

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def_rewrite_uses(
      &intr->def,
      nir_bindless_image_agx(b, nir_imul_imm(b, intr->src[0].ssa, 32),
                             .desc_set = 0));
   return true;
}

/* Mirror agx_build_background_shader for reloads followed by clears. */
static bool
compile_background(const struct agx_tilebuffer_layout *layout,
                   unsigned reloads, bool mrt_bg, unsigned reserved_preamble,
                   struct agx_shader_part *part)
{
   struct agx_tilebuffer_layout tib = *layout;
   unsigned count = 0;
   while (count < AGX_MAX_RENDER_TARGETS &&
          tib.logical_format[count] != PIPE_FORMAT_NONE)
      count++;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P mixed BG");
   b.shader->info.fs.untyped_color_outputs = true;
   for (unsigned rt = 0; rt < count; ++rt) {
      const unsigned nr = util_format_get_nr_components(tib.logical_format[rt]);
      nir_def *value;
      if (rt < reloads) {
         nir_def *coord = nir_u2u32(&b, nir_load_pixel_coord(&b));
         nir_def *tex = nir_txf(&b, coord, .texture_index = rt * 2,
                                .dim = GLSL_SAMPLER_DIM_2D,
                                .dest_type = nir_type_uint32,
                                .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP);
         value = nir_trim_vector(&b, tex, nr);
      } else {
         value = nir_load_preamble(&b, nr, 32, 4 + rt * 8);
      }
      nir_store_output(&b, value, nir_imm_int(&b, 0),
                       .write_mask = BITFIELD_MASK(nr),
                       .src_type = nir_type_uint32,
                       .io_semantics.location = FRAG_RESULT_DATA0 + rt,
                       .io_semantics.num_slots = 1);
      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + rt);
   }

   agx_nir_lower_texture(b.shader);
   agx_preprocess_nir(b.shader);
   unsigned bindless_base = 0;
   NIR_PASS(_, b.shader, agx_nir_lower_tilebuffer, &tib, NULL, &bindless_base,
            NULL, NULL, false, true);
   NIR_PASS(_, b.shader, agx_nir_lower_monolithic_msaa, tib.nr_samples);
   NIR_PASS(_, b.shader, agx_nir_lower_multisampled_image_store);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass, lower_tex_handle_to_u0,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, b.shader, agx_nir_lower_g17p_tile_samples, tib.nr_samples);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.ignore_tib_dependencies = true,
      .g17p_mrt_bg = mrt_bg,
      .reserved_preamble = reserved_preamble,
      .no_preamble = true,
   };
   const bool ok = agx_compile_shader_nir(b.shader, &key, part);
   ralloc_free(b.shader);
   return ok;
}

static int
test_background_programs(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts) + ARRAY_SIZE(wide_layouts);
        ++i) {
      const struct layout *l = i < ARRAY_SIZE(layouts)
                                  ? &layouts[i]
                                  : &wide_layouts[i - ARRAY_SIZE(layouts)];
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
      memcpy(formats, l->formats, sizeof(l->formats));
      const struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, l->count, 1, false);

      for (unsigned reloads = 0; reloads <= l->count; ++reloads) {
         struct agx_shader_part part = {0};
         CHECK(compile_background(&tib, reloads, true,
                                  reloads < l->count ? 4 + 8 * l->count : 0,
                                  &part));
         CHECK(part.info.uses_txf == (reloads > 0));
         CHECK(part.info.push_count ==
               (reloads < l->count ? 4 + 8 * l->count : 0));
         CHECK(!part.info.scratch_size);
         CHECK(!part.info.agx3_colour_link.entry_emitted);
         free(part.binary);
      }
   }
   return 0;
}

static bool
compile_epilog_samples(unsigned count, const enum pipe_format *formats,
                       unsigned written, uint64_t partitions, bool additive,
                       bool write_z, unsigned samples,
                       struct agx_shader_part *part)
{
   const enum pipe_blendfactor dst =
      additive ? PIPE_BLENDFACTOR_ONE : PIPE_BLENDFACTOR_ZERO;
   struct agx_fs_epilog_key epilog = {
      .link.loc_written = written,
      .link.size_32 = written,
      .link.write_z = write_z,
      .nr_samples = samples,
   };
   memset(epilog.remap, -1, sizeof(epilog.remap));
   for (unsigned rt = 0; rt < count; ++rt) {
      epilog.remap[rt] = rt;
      epilog.rt_formats[rt] = formats[rt];
      epilog.blend.rt[rt].colormask =
         BITFIELD_MASK(util_format_get_nr_components(formats[rt]));
      epilog.blend.rt[rt].mode = agx_pack_blend_standard(
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, dst, PIPE_BLEND_ADD,
         PIPE_BLENDFACTOR_ONE, dst);
   }
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &epilog);

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P MRT epilog");
   agx_nir_fs_epilog(&b, &epilog);
   agx_nir_lower_g17p_tile_samples(b.shader, samples);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {.dev.encoding = AGX_ISA_ENCODING_AGX3B,
                                .fs.nr_samples = samples,
                                .g17p_tile_partitions = partitions,
                                .secondary = true};
   const bool ok = agx_compile_shader_nir(b.shader, &key, part);
   ralloc_free(b.shader);
   return ok;
}

static bool
compile_epilog_written(unsigned count, const enum pipe_format *formats,
                       unsigned written, uint64_t partitions, bool additive,
                       bool write_z, struct agx_shader_part *part)
{
   return compile_epilog_samples(count, formats, written, partitions,
                                 additive, write_z, 1, part);
}

static bool
compile_epilog(unsigned count, const enum pipe_format *formats, bool additive,
               bool write_z, struct agx_shader_part *part)
{
   return compile_epilog_written(count, formats, BITFIELD_MASK(count), 0,
                                 additive, write_z, part);
}

/* Find each pixel instruction of a native witness, in order. */
static bool
contains_in_order(const struct agx_shader_part *part,
                  const uint8_t (*ops)[6], unsigned count)
{
   const uint8_t *text = part->binary;
   size_t at = 0;
   for (unsigned i = 0; i < count; ++i) {
      while (at + 6 <= part->info.binary_size &&
             memcmp(text + at, ops[i], 6))
         at++;
      if (at + 6 > part->info.binary_size)
         return false;
      at += 6;
   }
   return true;
}

/* The pixel transactions of Apple's store-only MRT fragments for these
 * layouts, from metal-nt render pipelines with the same colour formats and a
 * [[depth(any)]] output: the entry, the class-1 depth transaction, then the
 * colour partitions exactly as without depth. */
static const uint8_t entry_512[6] = {0x87, 0x02, 0x54, 0x00, 0x06, 0x00};
static const uint8_t wait_1[6] = {0x87, 0x02, 0x54, 0x01, 0x00, 0x00};
static const uint8_t depthf[6] = {0xd7, 0x14, 0x54, 0x00, 0x00, 0x03};
static const uint8_t release_1[6] = {0x07, 0x02, 0x54, 0x01, 0x00, 0x00};
static const uint8_t wait_192[6] = {0x87, 0x02, 0x54, 0xc0, 0x08, 0x00};
static const uint8_t release_192[6] = {0x07, 0x02, 0x54, 0xc0, 0x00, 0x00};
static const uint8_t wait_48[6] = {0x87, 0x02, 0x54, 0x30, 0x08, 0x00};
static const uint8_t release_48[6] = {0x07, 0x02, 0x54, 0x30, 0x00, 0x00};
static const uint8_t wait_12[6] = {0x87, 0x02, 0x54, 0x0c, 0x08, 0x00};
static const uint8_t release_12[6] = {0x07, 0x02, 0x54, 0x0c, 0x00, 0x00};
static const uint8_t release_524[6] = {0x07, 0x02, 0x54, 0x0c, 0x02, 0x00};
static const uint8_t release_560[6] = {0x07, 0x02, 0x54, 0x30, 0x02, 0x00};
static const uint8_t release_704[6] = {0x07, 0x02, 0x54, 0xc0, 0x02, 0x00};

/* lstcol f13 of each colour store, in store order. */
static unsigned
store_f13_mask(const struct agx_shader_part *part, unsigned *stores)
{
   const uint8_t *text = part->binary;
   unsigned mask = 0, n = 0;
   for (size_t at = 0; at + 12 <= part->info.binary_size;) {
      if (text[at] == 0xe7 && (text[at + 1] & 0x07) == 0x06 &&
          (text[at + 2] & 0xfc) == 0x54) {
         mask |= ((text[at + 7] >> 7) & 1) << n++;
         at += 12;
      } else {
         at++;
      }
   }
   *stores = n;
   return mask;
}

static int
test_epilogs(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts) + ARRAY_SIZE(wide_layouts);
        ++i) {
      const struct layout *l = i < ARRAY_SIZE(layouts)
                                  ? &layouts[i]
                                  : &wide_layouts[i - ARRAY_SIZE(layouts)];
      for (unsigned write_z = 0; write_z < 2; ++write_z) {
         struct agx_shader_part part = {0};
         CHECK(compile_epilog(l->count, l->formats, false, write_z, &part));
         CHECK(part.info.agx3_colour_link.colour_classes == 7);
         CHECK(part.info.agx3_colour_link.entry_emitted);

         unsigned n = 0;
         uint8_t sequence[16][6];
         memcpy(sequence[n++], entry_512, 6);
         if (write_z) {
            memcpy(sequence[n++], wait_1, 6);
            memcpy(sequence[n++], depthf, 6);
            memcpy(sequence[n++], release_1, 6);
         }
         if (l->count == 3) {
            memcpy(sequence[n++], wait_192, 6);
            memcpy(sequence[n++], release_192, 6);
            memcpy(sequence[n++], wait_48, 6);
            memcpy(sequence[n++], release_48, 6);
            memcpy(sequence[n++], wait_12, 6);
            memcpy(sequence[n++], release_524, 6);
         } else {
            memcpy(sequence[n++], wait_12, 6);
            memcpy(sequence[n++], wait_192, 6);
            memcpy(sequence[n++], release_192, 6);
            memcpy(sequence[n++], wait_48, 6);
            memcpy(sequence[n++], release_48, 6);
            memcpy(sequence[n++], release_524, 6);
         }
         CHECK(contains_in_order(&part, (const uint8_t (*)[6])sequence, n));

         /* Wider than five targets, the leading stores set f13. */
         unsigned stores;
         const unsigned f13 = store_f13_mask(&part, &stores);
         CHECK(stores == l->count);
         CHECK(f13 == BITFIELD_MASK(MAX2(l->count, 5) - 5));
         free(part.binary);
      }
   }
   return 0;
}

/* The pixel transactions of Apple's store-only fragments writing some of the
 * colour targets, from metal-nt -arch applegpu_g17p render pipelines of two to
 * four targets for every written subset, indexed by the written partitions.
 * Each written partition keeps the classes of its position among all targets,
 * partitions store in descending order, the last store also releases class
 * 512, and class 12 stays held from partition 3 to partition 0 only when both
 * are written. A lone store to partition 0 is the one-target protocol.
 */
enum pixel_op { END, W12, W48, W192, R12, R48, R192, R524, R560, R704 };

static const struct {
   unsigned count, written;
   enum pixel_op ops[8];
} subset_witnesses[] = {
   {3, 0x1, {W12, R524}},
   {3, 0x2, {W48, R560}},
   {3, 0x4, {W192, R704}},
   {3, 0x3, {W48, R48, W12, R524}},
   {3, 0x5, {W192, R192, W12, R524}},
   {3, 0x6, {W192, R192, W48, R560}},
   {4, 0x1, {W12, R524}},
   {4, 0x2, {W48, R560}},
   {4, 0x4, {W192, R704}},
   {4, 0x8, {W12, R524}},
   {4, 0x3, {W48, R48, W12, R524}},
   {4, 0x5, {W192, R192, W12, R524}},
   {4, 0x9, {W12, R524}},
   {4, 0x6, {W192, R192, W48, R560}},
   {4, 0xa, {W12, R12, W48, R560}},
   {4, 0xc, {W12, R12, W192, R704}},
   {4, 0x7, {W192, R192, W48, R48, W12, R524}},
   {4, 0xb, {W12, W48, R48, R524}},
   {4, 0xd, {W12, W192, R192, R524}},
   {4, 0xe, {W12, R12, W192, R192, W48, R560}},
};

/* The same for eight RGBA8 targets, from metal-nt -arch applegpu_g17p render
 * pipelines writing these subsets, with the number of leading stores that set
 * lstcol f13: every partition from 3 on shares class 12 with partition 0, and
 * a transaction storing more than five targets including 1 and 2 sets f13 on
 * all but its last five stores. */
static const struct {
   unsigned written, f13;
   enum pixel_op ops[12];
} octa_subset_witnesses[] = {
   {0x01, 0, {W12, R524}},
   {0x02, 0, {W48, R560}},
   {0x08, 0, {W12, R524}},
   {0x80, 0, {W12, R524}},
   {0x0c, 0, {W12, R12, W192, R704}},
   {0x19, 0, {W12, R524}},
   {0x21, 0, {W12, R524}},
   {0x46, 0, {W12, R12, W192, R192, W48, R560}},
   {0x88, 0, {W12, R524}},
   {0x8e, 0, {W12, R12, W192, R192, W48, R560}},
   {0x87, 0, {W12, W192, R192, W48, R48, R524}},
   {0xc7, 0, {W12, W192, R192, W48, R48, R524}},
   {0xf0, 0, {W12, R524}},
   {0x1f, 0, {W12, W192, R192, W48, R48, R524}},
   {0x3e, 0, {W12, R12, W192, R192, W48, R560}},
   {0xe6, 0, {W12, R12, W192, R192, W48, R560}},
   {0xf1, 0, {W12, R524}},
   {0xf8, 0, {W12, R524}},
   {0xf9, 0, {W12, R524}},
   {0xfa, 0, {W12, R12, W48, R560}},
   {0xfb, 0, {W12, W48, R48, R524}},
   {0xfc, 0, {W12, R12, W192, R704}},
   {0xfd, 0, {W12, W192, R192, R524}},
   {0x3f, 1, {W12, W192, R192, W48, R48, R524}},
   {0xe7, 1, {W12, W192, R192, W48, R48, R524}},
   {0xf6, 1, {W12, R12, W192, R192, W48, R560}},
   {0xf7, 2, {W12, W192, R192, W48, R48, R524}},
   {0xfe, 2, {W12, R12, W192, R192, W48, R560}},
   {0xff, 3, {W12, W192, R192, W48, R48, R524}},
};

/* Layouts whose one-channel 32-bit targets reach the epilog in their fixed
 * export registers, GPR half 8 + 8n, beyond a lone store's register bank:
 * dEQP-VK.renderpasses attachment.3.286, 4.430 and 4.420. */
static const struct layout scalar_layouts[] = {
   {"3.286", 3,
    {PIPE_FORMAT_R16_SINT, PIPE_FORMAT_R16_UINT, PIPE_FORMAT_R32_UINT},
    {4, 6, 0}, 8},
   {"4.430", 4,
    {PIPE_FORMAT_R8G8_SINT, PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32G32_UINT,
     PIPE_FORMAT_R8G8B8A8_UNORM},
    {12, 0, 4, 14}, 24},
   {"4.420", 4,
    {PIPE_FORMAT_R16G16B16A16_SINT, PIPE_FORMAT_R8G8B8A8_UNORM,
     PIPE_FORMAT_B10G10R10A2_UNORM, PIPE_FORMAT_R32_UINT},
    {8, 16, 0, 4}, 24},
};

static int
check_subset_epilogs(const struct layout *l, unsigned samples)
{
   static const uint8_t *const bytes[] = {
      [W12] = wait_12,     [W48] = wait_48,     [W192] = wait_192,
      [R12] = release_12,  [R48] = release_48,  [R192] = release_192,
      [R524] = release_524, [R560] = release_560, [R704] = release_704,
   };

   uint64_t partitions = 0;
   for (unsigned rt = 0; rt < l->count; ++rt)
      partitions |= BITFIELD64_BIT(l->offsets[rt]);

   for (unsigned written = 1; written < BITFIELD_MASK(l->count); ++written) {
      unsigned ranks = 0;
      u_foreach_bit(rt, written) {
         ranks |= BITFIELD_BIT(
            util_bitcount64(partitions & BITFIELD64_MASK(l->offsets[rt])));
      }

      unsigned w = 0;
      while (subset_witnesses[w].count != l->count ||
             subset_witnesses[w].written != ranks)
         w++;

      unsigned classes = 0, n = 0;
      u_foreach_bit(rank, ranks)
         classes |= BITFIELD_BIT(rank % 3);
      uint8_t sequence[8][6];
      for (; n < 8 && subset_witnesses[w].ops[n] != END; ++n)
         memcpy(sequence[n], bytes[subset_witnesses[w].ops[n]], 6);

      struct agx_shader_part part = {0};
      CHECK(compile_epilog_samples(l->count, l->formats, written, partitions,
                                   false, false, samples, &part));
      CHECK(contains_in_order(&part, (const uint8_t (*)[6])sequence, n));
      CHECK(part.info.agx3_colour_link.colour_classes == classes);
      free(part.binary);
   }
   return 0;
}

/* Apple's 2x/4x store-only fragments of every target acquire all their
 * classes in one PIXWAIT and release them once around stores through the
 * coverage register, with no lstcol f13 even for eight targets; fragments
 * writing only some targets bracket each partition exactly as at one sample
 * (metal-nt -arch applegpu_g17p render pipelines of two to eight RGBA8
 * targets, and of targets 1, 2, 1-2 of three and 1, 3 of four at 1x, 2x and
 * 4x: msaa/native/persample/{msmrt,msmrt2,mssubset}.py). */
static const uint8_t wait_60[6] = {0x87, 0x02, 0x54, 0x3c, 0x08, 0x00};
static const uint8_t wait_252[6] = {0x87, 0x02, 0x54, 0xfc, 0x08, 0x00};
static const uint8_t release_572[6] = {0x07, 0x02, 0x54, 0x3c, 0x02, 0x00};
static const uint8_t release_764[6] = {0x07, 0x02, 0x54, 0xfc, 0x02, 0x00};

static int
test_multisample_epilogs(void)
{
   for (unsigned samples = 2; samples <= 4; samples += 2) {
      for (unsigned i = 0; i < ARRAY_SIZE(layouts) + ARRAY_SIZE(wide_layouts);
           ++i) {
         const struct layout *l = i < ARRAY_SIZE(layouts)
                                     ? &layouts[i]
                                     : &wide_layouts[i - ARRAY_SIZE(layouts)];
         uint8_t sequence[3][6];
         memcpy(sequence[0], entry_512, 6);
         memcpy(sequence[1], l->count == 2 ? wait_60 : wait_252, 6);
         memcpy(sequence[2], l->count == 2 ? release_572 : release_764, 6);

         /* A layout whose tile would spill at this count is not stored. */
         enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
         memcpy(formats, l->formats, sizeof(l->formats));
         struct agx_tilebuffer_layout tib =
            agx_build_tilebuffer_layout(formats, l->count, samples, false);
         if (agx_tilebuffer_spills(&tib))
            continue;

         struct agx_shader_part part = {0};
         CHECK(compile_epilog_samples(l->count, l->formats,
                                      BITFIELD_MASK(l->count), 0, false, false,
                                      samples, &part));
         CHECK(contains_in_order(&part, (const uint8_t (*)[6])sequence, 3));
         unsigned stores = 0;
         CHECK(store_f13_mask(&part, &stores) == 0 && stores == l->count);
         free(part.binary);
      }

      for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i) {
         enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
         memcpy(formats, layouts[i].formats, sizeof(layouts[i].formats));
         struct agx_tilebuffer_layout tib = agx_build_tilebuffer_layout(
            formats, layouts[i].count, samples, false);
         if (!agx_tilebuffer_spills(&tib) &&
             check_subset_epilogs(&layouts[i], samples))
            return 1;
      }
   }
   return 0;
}

static int
test_octa_subset_epilogs(void)
{
   static const uint8_t *const bytes[] = {
      [W12] = wait_12,     [W48] = wait_48,     [W192] = wait_192,
      [R12] = release_12,  [R48] = release_48,  [R192] = release_192,
      [R524] = release_524, [R560] = release_560, [R704] = release_704,
   };
   const struct layout *l = &wide_layouts[3];
   CHECK(l->count == 8 && l->formats[7] == RGBA8);

   uint64_t partitions = 0;
   for (unsigned rt = 0; rt < l->count; ++rt)
      partitions |= BITFIELD64_BIT(l->offsets[rt]);

   for (unsigned w = 0; w < ARRAY_SIZE(octa_subset_witnesses); ++w) {
      const unsigned written = octa_subset_witnesses[w].written;
      unsigned classes = 0, n = 0;
      u_foreach_bit(rt, written)
         classes |= BITFIELD_BIT(rt < 3 ? rt : 0);
      uint8_t sequence[12][6];
      for (; n < 12 && octa_subset_witnesses[w].ops[n] != END; ++n)
         memcpy(sequence[n], bytes[octa_subset_witnesses[w].ops[n]], 6);

      struct agx_shader_part part = {0};
      CHECK(compile_epilog_written(l->count, l->formats, written,
                                   written == 0xff ? 0 : partitions, false,
                                   false, &part));
      CHECK(contains_in_order(&part, (const uint8_t (*)[6])sequence, n));
      CHECK(part.info.agx3_colour_link.colour_classes == classes);
      unsigned stores;
      CHECK(store_f13_mask(&part, &stores) ==
            BITFIELD_MASK(octa_subset_witnesses[w].f13));
      CHECK(stores == util_bitcount(written));
      free(part.binary);
   }
   return 0;
}

static int
test_subset_epilogs(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i) {
      if (check_subset_epilogs(&layouts[i], 1))
         return 1;
   }

   for (unsigned i = 0; i < ARRAY_SIZE(scalar_layouts); ++i) {
      const struct layout *l = &scalar_layouts[i];
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
      memcpy(formats, l->formats, sizeof(l->formats));
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, l->count, 1, false);
      CHECK(tib.sample_size_B == l->sample_size_B);
      for (unsigned rt = 0; rt < l->count; ++rt)
         CHECK(agx_tilebuffer_offset_B(&tib, rt) == l->offsets[rt]);
      if (check_subset_epilogs(l, 1))
         return 1;
   }
   return 0;
}

/* Homogeneous and two-target layouts keep their programs: sizes and FNV-1a
 * hashes of each binary as origin/agx3 2966c53569e compiled them. Blending
 * is covered where those layouts admit it. */
struct golden_program {
   uint32_t size;
   uint64_t hash;
};

static const struct {
   const char *name;
   unsigned count;
   enum pipe_format format0, other_formats;
   unsigned clear_push_halfs;
   struct golden_program programs[5];
} homogeneous[] = {
   {"rgba8x1", 1, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_NONE, 12,
    {{54, 0x4046fbb80be63de7ull},
     {176, 0x5f0f2b44b4423126ull},
     {36, 0x44e827688295a342ull},
     {94, 0xdac760eb596de1ceull},
     {68, 0xc35f74153a6f5d4dull}}},
   {"rgba16fx1", 1, PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_NONE, 12,
    {{58, 0x9bb0d0a9dc901048ull},
     {124, 0xaacc42d2bae1faf7ull},
     {56, 0x13fce0f2bc037095ull},
     {98, 0x2d5b1f727d37fe35ull},
     {68, 0x191002cafff501edull}}},
   {"rgba8x2", 2, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM, 20,
    {{98, 0xdb4a9c1199db533eull},
     {342, 0xbbaff21bc69ed97aull},
     {98, 0x3d556eefb6d79da2ull},
     {190, 0x537d1f57e7f03c12ull},
     {120, 0x1c83deaa038a3135ull}}},
   {"rgba8x3", 3, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM, 28,
    {{142, 0x219beb85e9dcef53ull},
     {0, 0},
     {100, 0x7e4ec8188383fa60ull},
     {226, 0x777d88deacbc5f85ull},
     {164, 0x3a6773eb9825a63full}}},
   {"rgba8x4", 4, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM, 36,
    {{174, 0xeca6c1a1fffa8aa6ull},
     {0, 0},
     {132, 0x254d0fd1f930dc33ull},
     {292, 0xe2f43523620bc947ull},
     {208, 0x343085049a8f7fd5ull}}},
   {"rgba32fx2", 2, PIPE_FORMAT_R32G32B32A32_FLOAT,
    PIPE_FORMAT_R32G32B32A32_FLOAT, 20,
    {{138, 0x8d1aa0edcf65a202ull},
     {208, 0x54f2d78621dd6b72ull},
     {60, 0x0acdf209897c54bbull},
     {168, 0xa6f1af8b4212793dull},
     {120, 0x1aa59cf1e9af7435ull}}},
   {"rgba32fx3", 3, PIPE_FORMAT_R32G32B32A32_FLOAT,
    PIPE_FORMAT_R32G32B32A32_FLOAT, 28,
    {{202, 0xed0c467782f8f8e0ull},
     {0, 0},
     {88, 0xb10acac55bc69275ull},
     {214, 0x90e525c9e03e950bull},
     {164, 0x41bfe3fe09b22dffull}}},
   {"rgba32ux4", 4, PIPE_FORMAT_R32G32B32A32_UINT,
    PIPE_FORMAT_R32G32B32A32_UINT, 36,
    {{254, 0x0a0b4e4dce2827feull},
     {0, 0},
     {116, 0x5d443c6410799bfbull},
     {260, 0xd2771c02d344217full},
     {208, 0x866cf34c0d0fbd55ull}}},
   {"r32fx4", 4, PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32_FLOAT, 36,
    {{104, 0x6b76264af3ddd455ull},
     {0, 0},
     {68, 0x9780d74361183b43ull},
     {212, 0x089fc7e5c4812c7full},
     {208, 0x866cf34c0d0fbd55ull}}},
   {"wboit", 2, PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R8_UNORM, 14,
    {{92, 0xf6185dff0bbeefa5ull},
     {208, 0x6c14b87b171d6988ull},
     {108, 0x6e9526fc4e86abbfull},
     {184, 0x84331807ec970b81ull},
     {238, 0x1815f518eb05705full}}},
   {"r11g11b10x2", 2, PIPE_FORMAT_R11G11B10_FLOAT,
    PIPE_FORMAT_R11G11B10_FLOAT, 18,
    {{366, 0x2d067c5b2fb41513ull},
     {648, 0xa42cad5cdc8d024cull},
     {442, 0xebed153ddb3c7d6dull},
     {458, 0x6226563426142c73ull},
     {120, 0x1aa59cf1e9af7435ull}}},
};

static int
check_golden(const char *name, bool ok, struct agx_shader_part *part,
             const struct golden_program *golden)
{
   if (!golden->size)
      return 0;

   if (!ok || part->info.binary_size != golden->size ||
       fnv1a(part->binary, part->info.binary_size) != golden->hash) {
      fprintf(stderr, "FAIL: %s changed from origin/agx3 (got {%u, 0x%016llxull})\n",
              name, ok ? part->info.binary_size : 0,
              ok ? (unsigned long long)fnv1a(part->binary,
                                             part->info.binary_size) : 0);
      return getenv("MRT_GOLDEN_ALL") ? 0 : 1;
   }

   checks++;
   free(part->binary);
   return 0;
}

static int
test_homogeneous_programs_unchanged(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(homogeneous); ++i) {
      const unsigned count = homogeneous[i].count;
      const struct golden_program *golden = homogeneous[i].programs;
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
      for (unsigned rt = 0; rt < count; ++rt)
         formats[rt] = rt ? homogeneous[i].other_formats
                          : homogeneous[i].format0;
      const struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, count, 1, false);
      CHECK(agx_g17p_classify_mrt(&tib) != AGX_G17P_MRT_MIXED);

      struct agx_shader_part part = {0};
      bool ok = compile_epilog(count, formats, false, false, &part);
      if (check_golden(homogeneous[i].name, ok, &part, &golden[0]))
         return 1;

      part = (struct agx_shader_part){0};
      ok = golden[1].size && compile_epilog(count, formats, true, false, &part);
      if (check_golden(homogeneous[i].name, ok, &part, &golden[1]))
         return 1;

      for (unsigned reload = 0; reload < 2; ++reload) {
         part = (struct agx_shader_part){0};
         ok = compile_background(&tib, reload ? count : 0, false,
                                 reload ? 0 : homogeneous[i].clear_push_halfs,
                                 &part);
         if (check_golden(homogeneous[i].name, ok, &part, &golden[2 + reload]))
            return 1;
      }

      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .no_preamble = true,
         .g17p_r11g11b10_mrt_eot =
            agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_R11G11B10,
      };
      agx_g17p_configure_rt0_eot_key(formats[0], &key);
      part = (struct agx_shader_part){0};
      ok = compile_eot(count, formats, tib._offset_B, &key, &part);
      if (check_golden(homogeneous[i].name, ok, &part, &golden[4]))
         return 1;
   }
   return 0;
}

int
main(void)
{
   if (test_selectors_match_single_target_programs() ||
       test_layout_classification() || test_eot_programs() ||
       test_background_programs() || test_epilogs() ||
       test_subset_epilogs() || test_octa_subset_epilogs() ||
       test_multisample_eot_witnesses() || test_multisample_epilogs() ||
       test_homogeneous_programs_unchanged())
      return 1;

   printf("G17P mixed-format MRT: %u checks PASS\n", checks);
   return 0;
}
