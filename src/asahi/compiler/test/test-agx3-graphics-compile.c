/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * End-to-end compile coverage for the AGX3 graphics stages.
 *
 * The AGX3 compute paths are covered by test-agx3-compile-fail-closed.c. This
 * file is the vertex/fragment counterpart, starting from a flat-shaded
 * triangle -- the smallest program that is still a real draw.
 *
 * The NIR here is written at the *post-lowering* level, i.e. what the backend
 * actually sees. agx_nir_lower_uvs() and agx_nir_lower_tilebuffer() live in
 * src/asahi/lib, which is built after the compiler and therefore cannot be
 * linked from a compiler test; store_uvs_agx / store_local_pixel_agx below are
 * exactly the intrinsics those two passes emit (agx_nir_lower_uvs.c:108,
 * agx_nir_lower_tilebuffer.c:75).
 *
 * Cases marked "expect reject" are gaps that remain fail-closed in
 * agx3_pack_binary(). They are asserted as rejections so that grounding one
 * later is a deliberate change to this table, not a silent behaviour drift.
 */

#include "agx_compile.h"
#include "agx3_copy_core.h"
#include "agx3_tex_core.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "asahi/lib/agx_abi.h"
#include "nir_builder.h"
#include "util/format/u_formats.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t agx3_end[] = {0x0e, 0x00, 0x00, 0x00};

/* Apple G17P's byte-exact colour-only tile protocol around the existing
 * normalized RGBA8 store. */
static const uint8_t g17p_flat_colour[] = {
   0x87, 0x02, 0x54, 0x00, 0x06, 0x00, /* entry gate */
   0x87, 0x02, 0x54, 0x0c, 0x08, 0x00, /* colour wait */
   0x97, 0x04, 0x54, 0x00, 0x02, 0x20, 0x80, 0x50, 0x04, 0xc8,
   0x97, 0x04, 0x54, 0x01, 0x02, 0x50, 0xc0, 0x50, 0x04, 0xc8,
   0xe7, 0x06, 0x54, 0x00, 0x00, 0x00, 0x01, 0x4e, 0x00, 0x00, 0x00, 0x00,
   0x07, 0x02, 0x54, 0x0c, 0x02, 0x00, /* final colour release */
   0x0e, 0x00, 0x00, 0x00,
};

/* Apple's complete G17P float2-position VS. UVSW operands name the separate
 * staging file, not general registers. The dataflow check below requires a
 * real producer in that file before accepting each position write. */
static const uint8_t apple_g17p_float2_vs[] = {
   0x0c, 0xdd, 0x10, 0x06,
   0x67, 0x10, 0x44, 0x00, 0x00, 0x80, 0x20, 0x00, 0x59, 0x01, 0x00, 0x40,
   0x48, 0x00,
   0x3b, 0x00, 0x29, 0xc4,
   0x2b, 0x03, 0x2f, 0x00, 0x42, 0x00, 0x00, 0x04, 0x00, 0x00,
   0x1b, 0x00, 0x26, 0x00, 0x40, 0x00, 0x00, 0x08, 0x00, 0x00,
   0x0c, 0x80, 0x23, 0x3e, 0x00, 0x00, 0x00, 0x0c, 0x40, 0x60,
   0x57, 0x16, 0x54, 0x06, 0x00, 0x40, 0x48, 0x00,
   0x57, 0x26, 0x54, 0x04, 0x20, 0x40, 0x49, 0x00,
   0x57, 0x46, 0x54, 0x02, 0x40, 0x40, 0x4a, 0x00,
   0x57, 0x86, 0x54, 0x00, 0x60, 0x40, 0x4b, 0x00,
   0x0e, 0x00, 0x00, 0x00,
};

static const uint8_t mesa_unstaged_g17p_float2_vs[] = {
   0x0c, 0xdd, 0x10, 0x06,
   0x67, 0x10, 0x44, 0x00, 0x00, 0x80, 0x20, 0x00, 0x59, 0x01, 0x00, 0x40,
   0x48, 0x00,
   0x57, 0x16, 0x54, 0x00, 0x00, 0x40, 0x48, 0x00,
   0x57, 0x26, 0x54, 0x02, 0x20, 0x40, 0x49, 0x00,
   0x0b, 0x00, 0x00, 0x00,
   0x57, 0x46, 0x54, 0x00, 0x40, 0x40, 0x4a, 0x00,
   0x0c, 0x80, 0x02, 0x3e, 0x00, 0x00, 0x00, 0x0c,
   0x57, 0x86, 0x54, 0x00, 0x60, 0x40, 0x4b, 0x00,
   0x0e, 0x00, 0x00, 0x00,
};

static const uint8_t mesa_g17p_float2_vs[] = {
   0x0c,0xdd,0x10,0x06,0x67,0x10,0x44,0x00,0x00,0x80,0x20,0x00,0x59,0x01,0x00,0x40,
   0x48,0x00,0x0b,0x00,0x21,0xc4,
   0x1b,0x03,0x27,0x00,0x42,0x00,0x00,0x04,0x00,0x00,
   0x0b,0x00,0x00,0x00,
   0x2b,0x01,0x27,0x00,0x42,0x00,0x00,0x08,0x00,0x00,
   0x0c,0x80,0x02,0x3e,0x00,0x00,0x00,0x0c,
   0x3b,0x01,0x27,0x00,0x42,0x00,0x00,0x0c,0x00,0x00,
   0x57,0x16,0x54,0x00,0x00,0x40,0x48,0x00,
   0x57,0x26,0x54,0x02,0x20,0x40,0x49,0x00,
   0x57,0x46,0x54,0x04,0x40,0x40,0x4a,0x00,
   0x57,0x86,0x54,0x06,0x60,0x40,0x4b,0x00,0x0e,0x00,0x00,0x00,
};

/* Exact post-lowering R16G16_SFLOAT vertex path.  The two six-byte f32add
 * instructions at +0x12/+0x18 are the corpus-grounded half-to-float widens;
 * both preserve their raw LDDEV halves through separate UVS exports. */
static const uint8_t mesa_g17p_half2_varying_vs[] = {
   0x0c,0xdd,0x10,0x06,0x67,0x10,0x54,0x02,0x01,0x00,0x20,0x00,0x49,0x01,0x00,0x40,
   0x44,0x00,0x29,0x02,0x1c,0x81,0x00,0xc2,0x39,0x02,0x1c,0x81,0x08,0x02,0x67,0x00,
   0x44,0x08,0x00,0x00,0x20,0x00,0x57,0x00,0x00,0x40,0x40,0x00,0x0b,0x08,0x21,0x44,
   0x1b,0x0b,0x27,0x00,0x42,0x00,0x00,0x04,0x00,0x00,0x2b,0x0d,0x27,0x00,0x42,0x00,
   0x00,0x08,0x00,0x00,0x3b,0x0f,0x27,0x00,0x42,0x00,0x00,0x0c,0x00,0x00,0x4b,0x05,
   0x27,0x00,0x42,0x00,0x00,0x10,0x00,0x00,0x5b,0x07,0x27,0x00,0x42,0x00,0x00,0x14,
   0x00,0x00,0x57,0x16,0x54,0x00,0x00,0x40,0x48,0x00,0x57,0x26,0x54,0x02,0x20,0x40,
   0x49,0x00,0x57,0x46,0x54,0x04,0x40,0x40,0x4a,0x00,0x57,0x86,0x54,0x06,0x60,0x40,
   0x4b,0x00,0x57,0x06,0x55,0x08,0x80,0x40,0x4c,0x00,0x57,0x06,0x56,0x0a,0xa0,0x40,
   0x4d,0x00,0x0e,0x00,0x00,0x00,
};

static bool
g17p_vs_position_dataflow(const uint8_t *binary, size_t size)
{
   if (size < 22 || memcmp(binary, apple_g17p_float2_vs, 18))
      return false;

   /* Symbolic values: 1=x, 2=y, 3=zero, 4=one. Decode only the grounded
    * producer forms occurring in these two isolated float2 shaders. Unlike
    * the former normalizer, this cannot mistake an unwritten staging slot
    * for a general register with the same numeric index. */
   uint8_t gpr[256] = {1, 0, 2}, staging[256] = {0};
   uint8_t staging_slot[256] = {0};
   bool attribute_ready = false;
   unsigned writes = 0, offset = 18;
   while (offset + 4 <= size) {
      const uint8_t *p = binary + offset;
      unsigned length = 0;
      const unsigned dest = (p[0] >> 4) * 2;
      if (p[0] == 0x0e && p[1] == 0 && p[2] == 0 && p[3] == 0)
         return writes == 4 && offset + 4 == size;
      if ((p[0] & 0xf) == 0xb &&
          ((p[2] & 0xf7) == 0x21 || (p[2] & 0xf7) == 0x27)) {
         length = (p[2] & 7) == 1 ? 4 : 10;
         if (length > size - offset)
            return false;
         /* Compact COPY has an enumerated one-hot wait and implicit slot1;
          * extended COPY has six independent wait bits and explicit f12. */
         const unsigned wait_index = p[3] >> 5;
         const unsigned wait = length == 4
                                  ? (wait_index ? 1u << (wait_index - 1) : 0)
                                  : (p[5] >> 5) | ((p[7] >> 5) << 3);
         attribute_ready |= (wait & 0x20) != 0;
         const unsigned src = p[1] & 0xfe;
         if ((gpr[src] == 1 || gpr[src] == 2) && !attribute_ready)
            return false;
         staging[dest] = gpr[src];
         staging_slot[dest] = length == 4 ? 1 : 1 + ((p[7] >> 2) & 7);
      } else if ((p[0] & 0xf) == 0xb && p[1] == 0 && p[2] == 0 &&
                 p[3] == 0) {
         length = 4;
         gpr[dest] = 3;
      } else if ((p[0] & 0xf) == 0xb && p[1] == 0 && p[2] == 0x26) {
         length = 10;
         if (length > size - offset)
            return false;
         staging[dest] = 3;
         staging_slot[dest] = 1 + ((p[7] >> 2) & 7);
      } else if ((p[0] & 0xf) == 0xc && p[1] == 0x80 &&
                 (p[2] == 2 || p[2] == 0x23) && p[3] == 0x3e) {
         length = p[2] == 2 ? 8 : 10;
         if (length > size - offset)
            return false;
         if (p[2] == 2)
            gpr[dest] = 4;
         else {
            staging[dest] = 4;
            staging_slot[dest] = 1 + (p[9] >> 5);
         }
      } else if (p[0] == 0x57 && p[2] == 0x54 && offset + 8 <= size) {
         length = 8;
         const unsigned component = p[4] >> 5;
         /* The matched native shape produces all four before first UVSW. */
         if (!staging[0] || !staging[2] || !staging[4] || !staging[6])
            return false;
         if (component != writes || staging[p[3]] != component + 1 ||
             (p[1] >> 4) != (1u << writes) ||
             staging_slot[p[3]] != 1 + (p[6] & 7) ||
             staging_slot[p[3]] != writes + 1)
            return false;
         writes++;
      } else {
         return false;
      }
      if (length > size - offset)
         return false;
      offset += length;
   }
   return false;
}

static bool
g17p_vs_matches_apple(const uint8_t *binary, size_t size)
{
   uint8_t bad_wait[sizeof(mesa_g17p_float2_vs)];
   memcpy(bad_wait, mesa_g17p_float2_vs, sizeof(bad_wait));
   bad_wait[21] = 4; /* staging COPY no longer waits for attribute LDDEV */
   uint8_t bad_bank[sizeof(mesa_g17p_float2_vs)];
   memcpy(bad_bank, mesa_g17p_float2_vs, sizeof(bad_bank));
   bad_bank[20] &= ~0x20; /* destination no longer selects staging */
   return !g17p_vs_position_dataflow(bad_wait, sizeof(bad_wait)) &&
          !g17p_vs_position_dataflow(bad_bank, sizeof(bad_bank)) &&
          g17p_vs_position_dataflow(apple_g17p_float2_vs,
                                    sizeof(apple_g17p_float2_vs)) &&
          !g17p_vs_position_dataflow(mesa_unstaged_g17p_float2_vs,
                                     sizeof(mesa_unstaged_g17p_float2_vs)) &&
          g17p_vs_position_dataflow(binary, size);
}

/* In 71 fully decoded AGX3b vertex programs, f79 selects byte2=0x44 on the
 * final LDDEV (and only the final LDDEV), while the first LDDEV always uses
 * scoreboard slot 6. Locate the compiler's 14-byte device-load forms and pin
 * those two corpus invariants without constraining allocation-only fields. */
static bool
g17p_vertex_load_policy(const uint8_t *binary, size_t size,
                        unsigned expected_loads)
{
   const uint8_t *loads[8];
   unsigned count = 0;

   for (unsigned i = 0; i + 14 <= size; ++i) {
      const uint8_t *p = binary + i;
      if (p[0] != 0x67 || (p[2] != 0x44 && p[2] != 0x54) ||
          p[6] != 0x20 || p[11] != 0x40 || p[13] != 0)
         continue;
      if (count == ARRAY_SIZE(loads))
         return false;
      loads[count++] = p;
      i += 13;
   }

   if (count != expected_loads)
      return false;

   for (unsigned i = 0; i < count; ++i) {
      const bool terminal = loads[i][2] == 0x44;
      if (terminal != (i + 1 == count))
         return false;
   }

   const unsigned first_slot =
      1 + (loads[0][8] >> 6) + 4 * (loads[0][9] & 1);
   return first_slot == 6;
}

/* AIL_ISA_FORMAT_U8NORM: the tilebuffer format for an RGBA8_UNORM target. */
#define AGX3_TEST_TIB_FORMAT PIPE_FORMAT_R8_UNORM

static const char *
enc_name(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3B ? "AGX3b" : "AGX3c";
}

/*
 * Minimal vertex shader: gl_Position = vec4(float(gl_VertexID), 0, 0, 1),
 * written to the UVS the way agx_nir_lower_uvs lowers a store_output of
 * VARYING_SLOT_POS with a single-varying layout (base index 0).
 */
static nir_shader *
build_vs_flat(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 flat triangle VS");

   nir_def *id = nir_u2f32(&b, nir_load_vertex_id(&b));
   nir_def *pos[4] = {
      id,
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 1.0f),
   };

   for (unsigned c = 0; c < 4; ++c)
      nir_store_uvs_agx(&b, pos[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written = VARYING_BIT_POS;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static nir_shader *
build_vs_g17p_div3_mul_high(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 div3 mul-high VS");
   nir_def *id = nir_load_vertex_id(&b);
   nir_def *high =
      nir_imul_high(&b, id, nir_imm_int(&b, UINT32_C(0x55555556)));
   nir_def *pos[4] = {
      nir_i2f32(&b, high),
      nir_u2f32(&b, id),
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 1.0f),
   };

   for (unsigned c = 0; c < 4; ++c)
      nir_store_uvs_agx(&b, pos[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written = VARYING_BIT_POS;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* Instance ID uses its own G17P special register. Keep a complete shader case
 * so the end-to-end compiler, not only the standalone rld encoder, proves the
 * hardware-vertex lowering. */
static nir_shader *
build_vs_instance_id(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 instance ID VS");

   nir_def *id = nir_u2f32(&b, nir_load_instance_id(&b));
   nir_def *pos[4] = {
      id,
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 1.0f),
   };

   for (unsigned c = 0; c < 4; ++c)
      nir_store_uvs_agx(&b, pos[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written = VARYING_BIT_POS;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/*
 * Vertex shader with a real attribute fetch: agx_nir_lower_vbo turns
 * load_input into the load_agx used here, so this is the shape a DRAW with a
 * vertex buffer actually reaches the backend with.
 */
static nir_shader *
build_vs_attribute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 attribute VS");

   nir_def *base = nir_imm_int64(&b, 0x1000);
   nir_def *index = nir_load_vertex_id(&b);
   nir_def *attr = nir_load_agx(&b, 4, 32, base, index, .access = 0, .base = 0,
                                .format = PIPE_FORMAT_R32_UINT,
                                .sign_extend = false);

   for (unsigned c = 0; c < 4; ++c) {
      nir_store_uvs_agx(&b, nir_channel(&b, attr, c),
                        nir_imm_intN_t(&b, c, 16));
   }

   b.shader->info.outputs_written = VARYING_BIT_POS;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* The live HAL200 vertex path fills uniform halfs 0..3 with the first VBO
 * address. This covers the same device load with that Ioto-populated base. */
static nir_shader *
build_vs_uniform_attribute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 uniform-base attribute VS");

   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *index = nir_load_vertex_id(&b);
   nir_def *attr = nir_load_agx(&b, 4, 32, base, index, .access = 0, .base = 0,
                                .format = PIPE_FORMAT_R32_UINT,
                                .sign_extend = false);

   for (unsigned c = 0; c < 4; ++c) {
      nir_store_uvs_agx(&b, nir_u2f32(&b, nir_channel(&b, attr, c)),
                        nir_imm_intN_t(&b, c, 16));
   }

   b.shader->info.outputs_written = VARYING_BIT_POS;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* The exact live G17P render-only shape: one float2 VBO attribute indexed by
 * vertex_id, expanded to a four-component position with z=0 and w=1. */
static nir_shader *
build_vs_g17p_float2_position(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G17P float2 position VS");

   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *index = nir_load_vertex_id(&b);
   nir_def *attr = nir_load_agx(&b, 2, 32, base, index, .access = 0, .base = 1,
                                .format = PIPE_FORMAT_R32_UINT,
                                .sign_extend = false);
   nir_def *position[] = {
      nir_channel(&b, attr, 0),
      nir_channel(&b, attr, 1),
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 1.0f),
   };

   for (unsigned c = 0; c < ARRAY_SIZE(position); ++c)
      nir_store_uvs_agx(&b, position[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written = VARYING_BIT_POS;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static nir_shader *
build_vs_g17p_single_instanced_varying(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options,
      "G17P single instanced load with varying VS");

   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *attr = nir_load_agx(&b, 2, 32, base, nir_load_instance_id(&b),
                                .access = 0, .base = 1,
                                .format = PIPE_FORMAT_R32_UINT,
                                .sign_extend = false);
   nir_def *values[] = {
      nir_channel(&b, attr, 0), nir_channel(&b, attr, 1),
      nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f),
      nir_channel(&b, attr, 0), nir_channel(&b, attr, 1),
   };

   for (unsigned c = 0; c < ARRAY_SIZE(values); ++c)
      nir_store_uvs_agx(&b, values[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static nir_shader *
build_vs_g17p_two_loads(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G17P two-load position VS");

   nir_def *index = nir_load_vertex_id(&b);
   nir_def *position = nir_load_agx(
      &b, 2, 32, nir_load_preamble(&b, 1, 64, .base = 0), index,
      .access = 0, .base = 1, .format = PIPE_FORMAT_R32_UINT,
      .sign_extend = false);
   nir_def *varying = nir_load_agx(
      &b, 2, 32, nir_load_preamble(&b, 1, 64, .base = 4), index,
      .access = 0, .base = 1, .format = PIPE_FORMAT_R32_UINT,
      .sign_extend = false);
   nir_def *values[] = {
      nir_channel(&b, position, 0), nir_channel(&b, position, 1),
      nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f),
      nir_channel(&b, varying, 0), nir_channel(&b, varying, 1),
   };

   for (unsigned c = 0; c < ARRAY_SIZE(values); ++c)
      nir_store_uvs_agx(&b, values[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* Post-VBO-lowering shape for a separate R16G16_SFLOAT attribute.  The raw
 * 32-bit LDDEV result carries two IEEE binary16 values; each is widened by
 * the native mixed-width f32add-with-negative-zero form before UVS export. */
static nir_shader *
build_vs_g17p_half2_varying(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G17P half2 varying VS");

   nir_def *vertex = nir_load_vertex_id(&b);
   nir_def *packed = nir_load_agx(
      &b, 1, 32, nir_load_preamble(&b, 1, 64, .base = 4), vertex,
      .access = 0, .base = 2 << AGX_NIR_MEMORY_PACKED_COUNT_SHIFT,
      .format = PIPE_FORMAT_R16_UINT, .sign_extend = false);
   nir_def *halfs[] = {
      nir_unpack_32_2x16_split_x(&b, packed),
      nir_unpack_32_2x16_split_y(&b, packed),
   };

   for (unsigned c = 0; c < ARRAY_SIZE(halfs); ++c)
      nir_store_uvs_agx(&b, nir_f2f32(&b, halfs[c]),
                        nir_imm_intN_t(&b, 4 + c, 16));

   nir_def *position = nir_load_agx(
      &b, 4, 32, nir_load_preamble(&b, 1, 64, .base = 0), vertex,
      .access = 0, .base = 2, .format = PIPE_FORMAT_R32_UINT,
      .sign_extend = false);
   for (unsigned c = 0; c < 4; ++c)
      nir_store_uvs_agx(&b, nir_channel(&b, position, c),
                        nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* Standard vkcube's vertex interface: position plus a vec4 texture coordinate
 * and a vec3 object-space position, occupying UVS words 4..10. */
static nir_shader *
build_vs_vkcube_varyings(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 vkcube varying VS");

   nir_def *id = nir_u2f32(&b, nir_load_vertex_id(&b));
   nir_def *values[11] = {
      id,
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 1.0f),
      id,
      nir_imm_float(&b, 0.25f),
      nir_imm_float(&b, 0.5f),
      nir_imm_float(&b, 1.0f),
      id,
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 1.0f),
   };

   for (unsigned c = 0; c < ARRAY_SIZE(values); ++c)
      nir_store_uvs_agx(&b, values[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0) |
      BITFIELD64_BIT(VARYING_SLOT_VAR1);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* The bounded live vkcube vertex shape: position and texture coordinates come
 * from two VBO CProg bases, while set 0 binding 0 supplies a column-major MVP
 * matrix through the descriptor-set pointer following the root pointer.
 */
static nir_shader *
build_vs_vkcube_vbo_ubo(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 vkcube VBO plus UBO VS");

   const unsigned nr_vbos = 2;
   const unsigned root = AGX_ABI_VUNI_COUNT_VK(nr_vbos);
   nir_def *vertex = nir_load_vertex_id(&b);
   nir_def *position = nir_load_agx(
      &b, 4, 32,
      nir_load_preamble(&b, 1, 64, .base = AGX_ABI_VUNI_VBO_BASE(0)), vertex,
      .access = 0, .base = 0, .format = PIPE_FORMAT_R32_UINT,
      .sign_extend = false);
   nir_def *texcoord = nir_load_agx(
      &b, 4, 32,
      nir_load_preamble(&b, 1, 64, .base = AGX_ABI_VUNI_VBO_BASE(1)), vertex,
      .access = 0, .base = 0, .format = PIPE_FORMAT_R32_UINT,
      .sign_extend = false);

   nir_def *set = nir_load_preamble(&b, 1, 64, .base = root + 4);
   nir_def *descriptor = nir_load_global_constant(
      &b, 4, 32, set, .align_mul = 16, .access = ACCESS_CAN_SPECULATE);
   nir_def *ubo = nir_pack_64_2x32_split(
      &b, nir_channel(&b, descriptor, 0), nir_channel(&b, descriptor, 1));

   nir_def *column[4];
   for (unsigned c = 0; c < ARRAY_SIZE(column); ++c) {
      column[c] = nir_load_global_constant(
         &b, 4, 32, nir_iadd_imm(&b, ubo, c * 16), .align_mul = 16,
         .access = ACCESS_CAN_SPECULATE);
   }

   nir_def *transformed[4];
   for (unsigned row = 0; row < ARRAY_SIZE(transformed); ++row) {
      transformed[row] = nir_ffma(
         &b, nir_channel(&b, position, 0), nir_channel(&b, column[0], row),
         nir_ffma(
            &b, nir_channel(&b, position, 1),
            nir_channel(&b, column[1], row),
            nir_ffma(&b, nir_channel(&b, position, 2),
                     nir_channel(&b, column[2], row),
                     nir_fmul(&b, nir_channel(&b, position, 3),
                              nir_channel(&b, column[3], row)))));
   }

   for (unsigned c = 0; c < 4; ++c) {
      nir_store_uvs_agx(&b, transformed[c], nir_imm_intN_t(&b, c, 16));
      nir_store_uvs_agx(&b, nir_channel(&b, texcoord, c),
                        nir_imm_intN_t(&b, 4 + c, 16));
   }
   for (unsigned c = 0; c < 3; ++c)
      nir_store_uvs_agx(&b, transformed[c], nir_imm_intN_t(&b, 8 + c, 16));

   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0) |
      BITFIELD64_BIT(VARYING_SLOT_VAR1);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/*
 * The FS lowering passes introduce driver-owned system values that every
 * driver resolves itself (d12_dxil_lower_root_mask / hk_lower_uvs_index and
 * friends). Resolve them to the monolithic-draw constants so the test
 * exercises the backend rather than a driver ABI.
 */
static bool
lower_driver_sysvals(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *_)
{
   uint16_t value;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_api_sample_mask_agx:
      value = 0xFF;
      break;
   case nir_intrinsic_load_shader_part_tests_zs_agx:
      value = 0;
      break;
   case nir_intrinsic_load_active_samples_agx:
      value = 0xFF;
      break;
   default:
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_intN_t(b, value, 16));
   return true;
}

static void
fs_lower(nir_shader *s)
{
   NIR_PASS(_, s, agx_nir_lower_discard_zs_emit);
   NIR_PASS(_, s, agx_nir_lower_sample_mask);
   NIR_PASS(_, s, agx_nir_lower_interpolation);
   NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_driver_sysvals,
            nir_metadata_control_flow, NULL);
}

static void
fs_store_colour(nir_builder *b, nir_def *colour)
{
   nir_store_local_pixel_agx(b, colour, nir_imm_intN_t(b, 0xFF, 16),
                             nir_undef(b, 2, 16), .base = 0, .write_mask = 0xf,
                             .format = AGX3_TEST_TIB_FORMAT);
}

static nir_shader *
fs_finish(nir_builder *b)
{
   b->shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_preprocess_nir(b->shader);

   fs_lower(b->shader);
   return b->shader;
}

static bool
test_g17p_background_clear(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P RT0 background clear");
   b.shader->info.internal = true;
   fs_store_colour(&b, nir_load_preamble(&b, 4, 32, .base = 4));
   nir_shader *nir = fs_finish(&b);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.ignore_tib_dependencies = true,
      .reserved_preamble = 12,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(nir, &key, &part);
   const bool ok = compiled && part.binary != NULL &&
                   part.info.binary_size > sizeof(agx3_end) &&
                   part.info.push_count == key.reserved_preamble &&
                   !part.info.has_preamble &&
                   memcmp((const uint8_t *)part.binary +
                             part.info.binary_size - sizeof(agx3_end),
                          agx3_end, sizeof(agx3_end)) == 0;
   if (!ok)
      fprintf(stderr, "AGX3b background-clear compiler mismatch\n");

   free(part.binary);
   ralloc_free(nir);
   return ok;
}

static bool
test_g17p_background_f16_clear_narrowing(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P f16 background clear");
   b.shader->info.internal = true;
   nir_store_local_pixel_agx(
      &b, nir_load_preamble(&b, 4, 32, .base = 4),
      nir_imm_intN_t(&b, 0xff, 16), nir_undef(&b, 2, 16), .base = 0,
      .write_mask = 0xf, .format = PIPE_FORMAT_R16_FLOAT);
   nir_shader *nir = fs_finish(&b);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.ignore_tib_dependencies = true,
      .reserved_preamble = 12,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(nir, &key, &part);
   static const uint8_t expected[] =
      "\x0b\x04\x01\x08\x01\x01\x1c\x81\x00\x02"
      "\x1b\x06\x01\x08\x01\x03\x1c\x81\x04\x02"
      "\x1b\x08\x01\x08\x11\x03\x1c\x81\x00\x02"
      "\x2b\x0a\x01\x08\x11\x05\x1c\x81\x04\x02"
      "\xe7\x06\x54\x00\x00\x00\x01\x0e\x00\x00\x00\x00"
      "\x0e\x00\x00\x00";
   bool ok = compiled && part.binary != NULL &&
             part.info.binary_size == sizeof(expected) - 1 &&
             !memcmp(part.binary, expected, sizeof(expected) - 1);
   if (!ok)
      fprintf(stderr, "AGX3b f16 background-clear narrowing mismatch\n");

   free(part.binary);
   ralloc_free(nir);

   return ok;
}

static bool
test_g17p_background_reload_staging(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P RT0 reload staging");
   b.shader->info.internal = true;
   nir_def *coord = nir_u2u32(&b, nir_load_pixel_coord(&b));
   nir_def *tex = nir_txf(&b, coord, .lod = nir_imm_int(&b, 0),
                          .dim = GLSL_SAMPLER_DIM_2D,
                          .dest_type = nir_type_float32,
                          .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP);
   fs_store_colour(&b, tex);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   nir_shader *nir = fs_finish(&b);
   NIR_PASS(_, nir, agx_nir_lower_texture);
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.ignore_tib_dependencies = true,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(nir, &key, &part);
   ok &= part.info.agx3_staging_known && part.info.agx3_staging_halfs == 4 &&
         part.info.uses_txf && part.info.push_count == 0 &&
         part.info.scratch_size == 0;
   unsigned found = 0;
   for (unsigned i = 8; part.binary && i + 14 <= part.info.binary_size; i++) {
      struct agx3_tex_fetch_desc fetch;
      if (!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B,
                                 (const uint8_t *)part.binary + i, 14, &fetch))
         continue;
      uint8_t x[AGX3_TEX_COPY_MAX_LENGTH], y[AGX3_TEX_COPY_MAX_LENGTH];
      uint8_t nx, ny;
      ok &= fetch.coord == 0 && fetch.wait_mask == (1u << (fetch.slot - 1)) &&
            agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, 4, fetch.slot, 0, x, &nx) &&
            agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 2, 6, fetch.slot, 0, y, &ny) &&
            i >= nx + ny &&
            !memcmp((const uint8_t *)part.binary + i - nx - ny, x, nx) &&
            !memcmp((const uint8_t *)part.binary + i - ny, y, ny);
      found++;
   }
   ok &= found == 1;
   if (!ok)
      fprintf(stderr, "G17P reload staging mismatch: fetches=%u known=%u halves=%u\n",
              found, part.info.agx3_staging_known, part.info.agx3_staging_halfs);
   free(part.binary);
   ralloc_free(nir);
   return ok;
}

/*
 * Minimal fragment shader: colour = vec4(0.25, 0.5, 0.75, 1.0) stored to
 * render target 0, as agx_nir_lower_tilebuffer lowers a store_output of
 * FRAG_RESULT_DATA0 into a single sample-replicating store_local_pixel_agx.
 */
static nir_shader *
build_fs_flat(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 flat triangle FS");

   fs_store_colour(&b, nir_vec4(&b, nir_imm_float(&b, 0.25f),
                                nir_imm_float(&b, 0.5f),
                                nir_imm_float(&b, 0.75f),
                                nir_imm_float(&b, 1.0f)));
   return fs_finish(&b);
}

/* Fragment shader reading a perspective-interpolated vec4 varying. */
static nir_shader *
build_fs_interpolated(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 interpolated FS");

   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *colour = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);

   fs_store_colour(&b, colour);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* Fragment shader with a discard: lowered to a sample_mask write. */
static nir_shader *
build_fs_discard(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 discard FS");

   nir_demote(&b);
   b.shader->info.fs.uses_discard = true;
   fs_store_colour(&b, nir_vec4(&b, nir_imm_float(&b, 0.25f),
                                nir_imm_float(&b, 0.5f),
                                nir_imm_float(&b, 0.75f),
                                nir_imm_float(&b, 1.0f)));
   return fs_finish(&b);
}

/* Fragment shader writing gl_FragDepth: lowered to zs_emit. */
static nir_shader *
build_fs_depth(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 depth-write FS");

   fs_store_colour(&b, nir_vec4(&b, nir_imm_float(&b, 0.25f),
                                nir_imm_float(&b, 0.5f),
                                nir_imm_float(&b, 0.75f),
                                nir_imm_float(&b, 1.0f)));
   nir_store_output(&b, nir_imm_float(&b, 0.5f), nir_imm_int(&b, 0),
                    .base = 0, .write_mask = 0x1, .src_type = nir_type_float32,
                    .io_semantics.location = FRAG_RESULT_DEPTH,
                    .io_semantics.num_slots = 1);

   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0) |
                                    BITFIELD64_BIT(FRAG_RESULT_DEPTH);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   return b.shader;
}

/*
 * Fragment shader that reads the tilebuffer before writing it -- the shape a
 * blend or a framebuffer fetch takes, and the shape Apple's own `w_pass8`
 * probe uses to pair a tile load's scoreboard slot with the store that
 * consumes it.
 */
static nir_shader *
build_fs_tilebuffer_load(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 tilebuffer-load FS");

   nir_def *old = nir_load_local_pixel_agx(
      &b, 4, 32, nir_imm_intN_t(&b, 0xFF, 16), .base = 0,
      .format = AGX3_TEST_TIB_FORMAT);
   fs_store_colour(&b, old);
   return fs_finish(&b);
}

/* Fragment shader matching vkcube's statically indexed set-0 combined image
 * sampler after Honeykrisp descriptor lowering. The Gen9 texture starts at
 * byte 64, followed by the native sampler word at byte 96.
 */
static nir_shader *
build_fs_texture_bias_offset_root(uint32_t bias_offset_B, bool use_root)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 textured FS");

   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *uv = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);

   nir_def *timage = nir_bindless_image_agx(&b, nir_imm_int(&b, 64),
                                            .desc_set = 0);
   nir_def *tsampler = nir_bindless_sampler_agx(
      &b, nir_imm_int(&b, 96), nir_imm_intN_t(&b, 45, 16), .desc_set = 0);

   /* Honeykrisp lowers the API sampler bias to a descriptor load. HAL200
    * carries that bias in the native sampler word, while the software field is
    * always zero, so the bounded lowering removes this source again.
    */
   nir_def *set = nir_load_descriptor_set_agx(&b, 0);
   nir_def *bias_offset = nir_u2u64(&b, nir_imm_int(&b, bias_offset_B));
   nir_def *bias = nir_load_global_constant(
      &b, 1, 16, nir_iadd(&b, set, bias_offset), .align_mul = 4,
      .align_offset = 0, .access = ACCESS_CAN_SPECULATE);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_txb;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, uv);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, timage);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, tsampler);
   tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_bias, bias);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   nir_def *colour = &tex->def;
   if (use_root) {
      nir_def *constant = nir_load_global_constant(
         &b, 1, 32, nir_load_root_agx(&b), .align_mul = 4,
         .access = ACCESS_CAN_SPECULATE);
      colour = nir_fmul(&b, colour, constant);
   }
   fs_store_colour(&b, colour);

   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   return b.shader;
}

static nir_shader *
build_fs_texture_bias_offset(uint32_t bias_offset_B)
{
   return build_fs_texture_bias_offset_root(bias_offset_B, false);
}

static bool
test_combined_sampler_root(void)
{
   nir_shader *shader = build_fs_texture_bias_offset_root(108, true);
   struct agx_g17p_combined_sampler_binding binding;
   bool ok = agx_nir_lower_g17p_combined_sampler(shader, &binding);
   unsigned roots = 0, descriptors = 0, samples = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               samples += tex->op == nir_texop_tex && tex->texture_index == 0 &&
                          tex->sampler_index == 0;
            } else if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
               roots += op == nir_intrinsic_load_root_agx;
               descriptors += op == nir_intrinsic_load_descriptor_set_agx ||
                              op == nir_intrinsic_bindless_image_agx ||
                              op == nir_intrinsic_bindless_sampler_agx;
            }
         }
      }
   }
   ok &= roots == 1 && descriptors == 0 && samples == 1 &&
         binding.texture_offset_B == 64 && binding.sampler_index_offset_B == 104;
   ralloc_free(shader);
   return ok;
}

static bool
test_indexed_sampler_sets(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "indexed sample sets");
   nir_def *root = nir_load_root_agx(&b);
   nir_def *element = nir_load_global_constant(&b, 1, 32, root, .align_mul=4);
   nir_def *offset = nir_imul_imm(&b, element, 72);
   nir_def *result = nir_imm_vec4(&b, 0, 0, 0, 0);
   const unsigned sets[] = {0, 3, 7};
   for (unsigned i = 0; i < ARRAY_SIZE(sets); ++i) {
      nir_def *address = nir_load_descriptor_set_agx(&b, sets[i]);
      address = nir_iadd(&b, address, nir_u2u64(&b, offset));
      nir_def *sampler_index = nir_load_global_constant(
         &b, 1, 16, nir_iadd_imm(&b, address, 40), .align_mul=2);
      nir_def *image = nir_bindless_image_agx(&b, offset, .desc_set=sets[i]);
      nir_def *sampler = nir_bindless_sampler_agx(
         &b, nir_iadd_imm(&b, offset, 32), sampler_index, .desc_set=sets[i]);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
      tex->op=nir_texop_tex; tex->sampler_dim=GLSL_SAMPLER_DIM_2D;
      tex->dest_type=nir_type_float32; tex->is_array=i==2;
      tex->coord_components=tex->is_array?3:2;
      tex->src[0]=nir_tex_src_for_ssa(
         nir_tex_src_coord,
         tex->is_array
            ? nir_vec3(&b, nir_imm_float(&b, .25), nir_imm_float(&b, .75),
                       nir_imm_int(&b, 1))
            : nir_imm_vec2(&b, .25, .75));
      tex->src[1]=nir_tex_src_for_ssa(nir_tex_src_texture_handle,image);
      tex->src[2]=nir_tex_src_for_ssa(nir_tex_src_sampler_handle,sampler);
      nir_def_init(&tex->instr,&tex->def,4,32);
      nir_builder_instr_insert(&b,&tex->instr);
      result=nir_fadd(&b,result,&tex->def);
   }
   nir_store_output(&b,result,nir_imm_int(&b,0),.write_mask=15,
                    .src_type=nir_type_float32);
   bool ok=agx_nir_lower_g17p_indexed_samplers(b.shader);
   unsigned samples=0, arrays=0, descriptors=0, roots=0, image_loads=0,
            sampler_loads=0, set_mask=0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr,block) {
         if (instr->type==nir_instr_type_tex) {
            nir_tex_instr *tex=nir_instr_as_tex(instr);
            int ti=nir_tex_instr_src_index(tex,nir_tex_src_texture_offset);
            int si=nir_tex_instr_src_index(tex,nir_tex_src_sampler_handle);
            ok &= ti>=0 && si>=0;
            if (ti>=0 && si>=0)
               ok &= nir_src_bit_size(tex->src[ti].src)==64 &&
                     nir_src_bit_size(tex->src[si].src)==64;
            samples++;
            arrays += tex->is_array;
         } else if (instr->type==nir_instr_type_intrinsic) {
            nir_intrinsic_instr *in=nir_instr_as_intrinsic(instr);
            roots += in->intrinsic==nir_intrinsic_load_root_agx;
            descriptors += in->intrinsic==nir_intrinsic_bindless_image_agx ||
                           in->intrinsic==nir_intrinsic_bindless_sampler_agx;
            if (in->intrinsic==nir_intrinsic_load_descriptor_set_agx)
               set_mask |= 1u << nir_intrinsic_desc_set(in);
            if (in->intrinsic==nir_intrinsic_load_global_constant) {
               image_loads += in->def.bit_size==32;
               sampler_loads += in->def.bit_size==16;
            }
         }
      }
   }
   ok &= samples==3 && arrays==1 && descriptors==0 && roots==1 && image_loads==4 &&
         sampler_loads==3 && set_mask==((1u<<0)|(1u<<3)|(1u<<7));
   ralloc_free(b.shader);
   return ok;
}

static bool
lower_test_descriptor_set(nir_builder *b, nir_intrinsic_instr *intr,
                          UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_descriptor_set_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_load_preamble(b, 1, 64,
                                     .base = 4 * nir_intrinsic_desc_set(intr)));
   return true;
}

static bool
test_g17p_cube_sample_lowering(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P cube sample lowering");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *coord = nir_load_interpolated_input(
      &b, 3, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *image =
      nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 0);
   nir_def *set = nir_load_descriptor_set_agx(&b, 0);
   nir_def *heap_index = nir_load_global_constant(
      &b, 1, 16, nir_iadd_imm(&b, set, 40), .align_mul = 2,
      .access = ACCESS_CAN_SPECULATE);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_imm_int(&b, 32), heap_index, .desc_set = 0);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 3;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   fs_store_colour(&b, &tex->def);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   bool ok = agx_nir_lower_g17p_cube_samples(b.shader) &&
             tex->sampler_dim == GLSL_SAMPLER_DIM_2D && tex->is_array &&
             tex->array_is_lowered_cube && tex->coord_components == 3;
   int ci = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
   ok &= ci >= 0 && nir_src_num_components(tex->src[ci].src) == 3 &&
         agx_nir_lower_g17p_indexed_samplers(b.shader);

   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
            lower_test_descriptor_set, nir_metadata_control_flow, NULL);

   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   key.reserved_preamble = 24;
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &key, &part) && part.binary != NULL;
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static nir_tex_instr *
custom_border_sample(nir_builder *b, nir_def *image, nir_def *sampler,
                     nir_def *coord, unsigned flags)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->backend_flags = flags;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return tex;
}

static bool
test_custom_border_samplers(bool dynamic, unsigned invalid)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "custom border indexed samples");
   nir_def *set = nir_load_descriptor_set_agx(&b, 24);
   nir_def *ordinary_index = dynamic ? nir_load_global_constant(
      &b, 1, 16, nir_iadd_imm(&b, set, 40), .align_mul = 2)
      : nir_imm_intN_t(&b, 80, 16);
   nir_def *clamp_index = dynamic ? nir_load_global_constant(
      &b, 1, 16, nir_iadd_imm(&b, set, 42), .align_mul = 2)
      : nir_imm_intN_t(&b, 84, 16);
   nir_def *image = nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 24);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_imm_int(&b, 32), ordinary_index, .desc_set = 24);
   nir_def *coord = nir_imm_vec2(&b, -.125, .75);
   nir_tex_instr *ordinary = custom_border_sample(&b, image, sampler, coord, 0);
   nir_def *condition = nir_ige_imm(&b, clamp_index, 0);
   nir_if *branch = nir_push_if(&b, condition);
   nir_tex_instr *clamp = custom_border_sample(
      &b, image, clamp_index, coord,
      invalid == 2 ? 0 :
      invalid ? AGX_TEXTURE_FLAG_NO_CLAMP : AGX_TEXTURE_FLAG_CLAMP_TO_0);
   nir_def *border = nir_load_global_constant(
      &b, 4, 32, nir_iadd_imm(&b, set, 52), .align_mul = 4);
   nir_def *mixed = nir_flrp(&b, &clamp->def, &ordinary->def, border);
   nir_pop_if(&b, branch);
   nir_def *result = nir_if_phi(&b, mixed, &ordinary->def);
   nir_store_output(&b, result, nir_imm_int(&b, 0), .write_mask = 15,
                    .src_type = nir_type_float32);

   bool lowered = agx_nir_lower_g17p_indexed_samplers(b.shader);
   bool ok = lowered == (invalid == 0);
   if (invalid) {
      /* An unsupported third shape must not partially convert the first. */
      ok &= nir_tex_instr_src_index(ordinary, nir_tex_src_texture_handle) >= 0;
      ok &= nir_tex_instr_src_index(clamp, nir_tex_src_texture_handle) >= 0;
   } else {
      ok &= branch->condition.ssa == condition;
      ok &= nir_def_instr(mixed)->block == clamp->instr.block;
      ok &= clamp->instr.block != ordinary->instr.block;
      nir_alu_instr *lerp = nir_instr_as_alu(nir_def_instr(mixed));
      ok &= lerp->op == nir_op_flrp && lerp->src[0].src.ssa == &clamp->def &&
            lerp->src[1].src.ssa == &ordinary->def &&
            lerp->src[2].src.ssa == border;
      nir_tex_instr *samples[] = {ordinary, clamp};
      nir_def *original[] = {ordinary_index, clamp_index};
      for (unsigned i = 0; i < 2; ++i) {
         nir_tex_instr *tex = samples[i];
         int si = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
         int ti = nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
         ok &= si >= 0 && ti >= 0;
         if (si < 0 || ti < 0)
            continue;
         ok &= nir_src_bit_size(tex->src[si].src) == 64 &&
               nir_src_bit_size(tex->src[ti].src) == 64;
         nir_alu_instr *wide = nir_def_as_alu_or_null(tex->src[si].src.ssa);
         nir_alu_instr *subtract = wide ? nir_def_as_alu_or_null(wide->src[0].src.ssa) : NULL;
         nir_alu_instr *extend = subtract ? nir_def_as_alu_or_null(subtract->src[0].src.ssa) : NULL;
         ok &= wide && wide->op == nir_op_u2u64 && subtract &&
               subtract->op == nir_op_iadd && extend &&
               extend->op == nir_op_u2u32 && extend->src[0].src.ssa == original[i];
      }
      if (!dynamic) {
         NIR_PASS(_, b.shader, nir_opt_constant_folding);
         for (unsigned i = 0; i < 2; ++i) {
            int si = nir_tex_instr_src_index(samples[i], nir_tex_src_sampler_handle);
            ok &= nir_src_is_const(samples[i]->src[si].src) &&
                  nir_src_as_uint(samples[i]->src[si].src) == (i ? 56 : 52);
         }
      }
   }
   ralloc_free(b.shader);
   return ok;
}

static bool
test_indexed_fetch_sets(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "mixed indexed sample and array fetch sets");
   nir_def *root = nir_load_root_agx(&b);
   nir_def *element = nir_load_global_constant(&b, 1, 32, root, .align_mul=4);
   nir_def *offset = nir_imul_imm(&b, element, 72);
   nir_def *result = nir_imm_vec4(&b, 0, 0, 0, 0);
   const unsigned sets[] = {0, 3, 7};
   for (unsigned i = 0; i < ARRAY_SIZE(sets); ++i) {
      nir_def *address = nir_load_descriptor_set_agx(&b, sets[i]);
      address = nir_iadd(&b, address, nir_u2u64(&b, offset));
      nir_def *sampler_index = nir_load_global_constant(
         &b, 1, 16, nir_iadd_imm(&b, address, 40), .align_mul=2);
      nir_def *image = nir_bindless_image_agx(&b, offset, .desc_set=sets[i]);
      nir_def *sampler = nir_bindless_sampler_agx(
         &b, nir_iadd_imm(&b, offset, 32), sampler_index, .desc_set=sets[i]);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
      tex->op=i ? nir_texop_txf : nir_texop_tex;
      tex->sampler_dim=GLSL_SAMPLER_DIM_2D;
      tex->dest_type=nir_type_float32;
      tex->is_array=i==2;
      tex->coord_components=i==1?1:(tex->is_array?3:2);
      tex->src[0]=nir_tex_src_for_ssa(i==1?nir_tex_src_backend1:nir_tex_src_coord,
         i ? (i==2 ? nir_imm_ivec3(&b,3,4,1) : nir_imm_ivec2(&b,3,4))
           : nir_imm_vec2(&b,.25,.75));
      tex->src[1]=nir_tex_src_for_ssa(nir_tex_src_texture_handle,image);
      tex->src[2]=nir_tex_src_for_ssa(i?nir_tex_src_lod:nir_tex_src_sampler_handle,
                                    i?nir_imm_intN_t(&b,0,16):sampler);
      nir_def_init(&tex->instr,&tex->def,4,32);
      nir_builder_instr_insert(&b,&tex->instr);
      result=nir_fadd(&b,result,&tex->def);
   }
   nir_store_output(&b,result,nir_imm_int(&b,0),.write_mask=15,
                    .src_type=nir_type_float32);
   bool ok=agx_nir_lower_g17p_indexed_samplers(b.shader);
   unsigned samples=0, fetches=0, arrays=0, descriptors=0, roots=0, image_loads=0, sampler_loads=0, set_mask=0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr,block) {
         if (instr->type==nir_instr_type_tex) {
            nir_tex_instr *tex=nir_instr_as_tex(instr);
            int ti=nir_tex_instr_src_index(tex,nir_tex_src_texture_offset);
            int si=nir_tex_instr_src_index(tex,nir_tex_src_sampler_handle);
            ok &= ti>=0;
            if(ti>=0)ok &= nir_src_bit_size(tex->src[ti].src)==64;
            if(tex->op==nir_texop_tex) {
               ok &= si>=0;
               if(si>=0)ok &= nir_src_bit_size(tex->src[si].src)==64;
               samples++;
            } else {
               ok &= si<0 && tex->op==nir_texop_txf;
               int li=nir_tex_instr_src_index(tex,nir_tex_src_lod);
               ok &= li>=0 && nir_src_is_const(tex->src[li].src) &&
                     nir_src_as_uint(tex->src[li].src)==0;
               fetches++;
               arrays+=tex->is_array;
            }
         } else if (instr->type==nir_instr_type_intrinsic) {
            nir_intrinsic_instr *in=nir_instr_as_intrinsic(instr);
            roots += in->intrinsic==nir_intrinsic_load_root_agx;
            descriptors += in->intrinsic==nir_intrinsic_bindless_image_agx ||
                           in->intrinsic==nir_intrinsic_bindless_sampler_agx;
            if (in->intrinsic==nir_intrinsic_load_descriptor_set_agx)
               set_mask |= 1u << nir_intrinsic_desc_set(in);
            if (in->intrinsic==nir_intrinsic_load_global_constant) {
               image_loads += in->def.bit_size==32;
               sampler_loads += in->def.bit_size==16;
            }
         }
      }
   }
   ok &= samples==1 && fetches==2 && arrays==1 && descriptors==0 && roots==1 && image_loads==4 &&
         sampler_loads==1 && set_mask==((1u<<0)|(1u<<3)|(1u<<7));
   ralloc_free(b.shader);
   return ok;
}

static nir_shader *
build_fs_texture(void)
{
   return build_fs_texture_bias_offset(108);
}

static bool
test_combined_sampler_scope(void)
{
   nir_shader *nir = build_fs_texture_bias_offset(112);
   struct agx_g17p_combined_sampler_binding binding = {
      .texture_offset_B = UINT32_MAX,
      .sampler_index_offset_B = UINT32_MAX,
   };

   const bool lowered =
      agx_nir_lower_g17p_combined_sampler(nir, &binding);
   const bool ok = !lowered && binding.texture_offset_B == UINT32_MAX &&
                   binding.sampler_index_offset_B == UINT32_MAX;
   ralloc_free(nir);
   return ok;
}

static nir_shader *
build_cs_bindless_txf(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 static bindless TXF CS");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *coord = nir_load_global_invocation_id(&b, 32);
   nir_def *handle = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 64), .desc_set = 0);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_array = true;
   tex->dest_type = nir_type_uint32;
   tex->coord_components = 3;
   tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(&b, 0));
   tex->src[2] =
      nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   /* Keep the fetch live without introducing an unrelated memory operation. */
   nir_def *sink = nir_channel(&b, &tex->def, 0);
   nir_export_agx(&b, sink, .base = 0);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static bool
test_txf_lowering(void)
{
   nir_shader *nir = build_cs_bindless_txf();
   struct agx_g17p_txf_binding binding = {.texture_offset_B = UINT32_MAX};

   const bool lowered = agx_nir_lower_g17p_txf(nir, &binding);
   bool ok = lowered && binding.texture_offset_B == 64;
   if (ok) {
      nir_tex_instr *fetch = NULL;
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_tex)
                  fetch = nir_instr_as_tex(instr);
            }
         }
      }

      ok = fetch && fetch->texture_index == 0 &&
           nir_tex_instr_src_index(fetch, nir_tex_src_texture_handle) < 0;
   }

   ralloc_free(nir);
   return ok;
}

static nir_shader *
build_cs_storage_image(unsigned mode)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 bounded storage image CS");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   const enum pipe_format format =
      mode == 1 ? PIPE_FORMAT_R16G16B16A16_UINT
                : PIPE_FORMAT_R8G8B8A8_UINT;
   const enum glsl_sampler_dim dim =
      mode == 2 ? GLSL_SAMPLER_DIM_3D
      : mode == 3 ? GLSL_SAMPLER_DIM_MS
                  : GLSL_SAMPLER_DIM_2D;
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *coord = nir_vec4(&b, nir_channel(&b, gid, 0),
                             nir_channel(&b, gid, 1), zero, zero);
   nir_def *texture = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 64), .desc_set = 0);
   nir_def *pbe = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 96), .desc_set = 0);
   if (mode == 4) {
      nir_def *descriptor = nir_load_from_texture_handle_agx(&b, texture);
      nir_def *width = nir_load_global_constant(
         &b, 1, 32, descriptor, .align_mul = 4);
      coord = nir_vec4(&b, nir_iand(&b, nir_channel(&b, gid, 0), width),
                       nir_channel(&b, gid, 1), zero, zero);
   }
   nir_def *value = nir_bindless_image_load(
      &b, 4, 32, texture, coord, zero, zero, .image_dim = dim,
      .image_array = false, .format = format, .access = ACCESS_RESTRICT);
   nir_bindless_image_store(&b, pbe, coord, zero, value, zero,
                            .image_dim = dim, .image_array = false,
                            .format = format, .access = ACCESS_RESTRICT);
   return b.shader;
}

static bool
test_storage_image_lowering(void)
{
   for (unsigned mode = 0; mode < 5; ++mode) {
      nir_shader *nir = build_cs_storage_image(mode);
      struct agx_g17p_storage_image_binding binding = {
         .texture_offset_B = UINT32_MAX,
         .pbe_offset_B = UINT32_MAX,
      };
      const bool lowered = agx_nir_lower_g17p_storage_image(nir, &binding);
      unsigned bindless_handles = 0, bindless_ops = 0;
      unsigned ordinary_loads = 0, ordinary_stores = 0;
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               const nir_intrinsic_op op =
                  nir_instr_as_intrinsic(instr)->intrinsic;
               bindless_handles += op == nir_intrinsic_bindless_image_agx;
               bindless_ops += op == nir_intrinsic_bindless_image_load ||
                               op == nir_intrinsic_bindless_image_store;
               ordinary_loads += op == nir_intrinsic_image_load;
               ordinary_stores += op == nir_intrinsic_image_store;
            }
         }
      }

      const bool ok = mode == 0 || mode == 4
         ? lowered && binding.texture_offset_B == 64 &&
              binding.pbe_offset_B == 96 &&
              bindless_handles == (mode == 4 ? 1 : 0) &&
              bindless_ops == 0 && ordinary_loads == 1 && ordinary_stores == 1
         : !lowered && binding.texture_offset_B == UINT32_MAX &&
              binding.pbe_offset_B == UINT32_MAX && bindless_handles == 2 &&
              bindless_ops == 2 && ordinary_loads == 0 && ordinary_stores == 0;
      ralloc_free(nir);
      if (!ok)
         return false;
   }

   return true;
}

static nir_shader *
build_cs_2d_image_access(bool store)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      store ? "AGX3 robust 2D image store" : "AGX3 robust 2D image load");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_variable *image = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT), "image");
   image->data.image.format = PIPE_FORMAT_R8G8B8A8_UINT;
   nir_def *deref = &nir_build_deref_var(&b, image)->def;
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *coord = nir_vec4(&b, nir_channel(&b, gid, 0),
                             nir_channel(&b, gid, 1), zero, zero);

   if (store) {
      nir_image_deref_store(
         &b, deref, coord, zero, nir_imm_ivec4(&b, 1, 2, 3, 4), zero,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
         .format = PIPE_FORMAT_R8G8B8A8_UINT, .access = ACCESS_RESTRICT,
         .src_type = nir_type_uint32);
   } else {
      nir_def *value = nir_image_deref_load(
         &b, 4, 32, deref, coord, zero, zero,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
         .format = PIPE_FORMAT_R8G8B8A8_UINT, .access = ACCESS_RESTRICT,
         .dest_type = nir_type_uint32);

      /* Keep the robust result live so the explicit zero select is part of
       * the test's observable graph. */
      nir_store_scratch(&b, value, nir_imm_int(&b, 0), .write_mask = 0xf,
                        .align_mul = 16);
   }

   return b.shader;
}

static bool
test_g17p_2d_robustness_scope(void)
{
   for (unsigned store = 0; store < 2; ++store) {
      nir_shader *nir = build_cs_2d_image_access(store);
      agx_nir_lower_texture_early_with_2d_robustness(nir, true, true);

      unsigned loads = 0, stores = 0, sizes = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_image_deref_load:
               loads++;
               break;
            case nir_intrinsic_image_deref_store:
               stores++;
               break;
            case nir_intrinsic_image_deref_size:
               sizes++;
               break;
            default:
               break;
            }
         }
      }

      const bool ok = store ? loads == 0 && stores == 1 && sizes == 0
                            : loads == 1 && stores == 0 && sizes == 1;
      ralloc_free(nir);
      if (!ok)
         return false;
   }

   return true;
}

static nir_shader *
build_cs_sampled_storage(unsigned mode)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "AGX3 sampled texture to storage image CS");
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 8;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *coord2 = nir_vec2(&b, nir_u2f32(&b, nir_channel(&b, gid, 0)),
                              nir_u2f32(&b, nir_channel(&b, gid, 1)));
   nir_def *texture = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 0), .desc_set = 0);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_imm_int(&b, 32), nir_imm_intN_t(&b, 28, 16), .desc_set = 0);
   nir_tex_instr *sample = nir_tex_instr_create(b.shader, 4);
   sample->op = nir_texop_txl;
   sample->sampler_dim = GLSL_SAMPLER_DIM_2D;
   sample->dest_type = nir_type_float32;
   sample->coord_components = 2;
   sample->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord2);
   sample->src[1] =
      nir_tex_src_for_ssa(nir_tex_src_texture_handle, texture);
   sample->src[2] =
      nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   sample->src[3] =
      nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 0.0f));
   nir_def_init(&sample->instr, &sample->def, 4, 32);
   nir_builder_instr_insert(&b, &sample->instr);

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *coord4 = nir_vec4(&b, nir_channel(&b, gid, 0),
                              nir_channel(&b, gid, 1), zero, zero);
   nir_def *pbe = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 32), .desc_set = 1);
   nir_def *value = mode == 2
      ? nir_fadd_imm(&b, &sample->def, 1.0f)
      : &sample->def;
   nir_bindless_image_store(
      &b, pbe, coord4, zero, value, zero,
      .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
      .format = mode == 1 ? PIPE_FORMAT_R8G8B8A8_UINT
                          : PIPE_FORMAT_R8G8B8A8_UNORM,
      .access = ACCESS_RESTRICT);
   return b.shader;
}

static bool
test_sampled_storage_lowering(void)
{
   for (unsigned mode = 0; mode < 3; ++mode) {
      nir_shader *nir = build_cs_sampled_storage(mode);
      struct agx_g17p_sampled_storage_binding binding = {
         .texture_offset_B = UINT32_MAX,
         .sampler_index_offset_B = UINT32_MAX,
         .pbe_offset_B = UINT32_MAX,
         .texture_set = UINT8_MAX,
         .pbe_set = UINT8_MAX,
      };
      const bool lowered =
         agx_nir_lower_g17p_sampled_storage(nir, &binding);
      unsigned bindless_handles = 0, bindless_stores = 0;
      unsigned ordinary_stores = 0, samples = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               samples++;
               if (lowered &&
                   (nir_tex_instr_src_index(tex,
                                            nir_tex_src_texture_handle) >= 0 ||
                    nir_tex_instr_src_index(tex,
                                            nir_tex_src_sampler_handle) >= 0))
                  samples += 100;
            } else if (instr->type == nir_instr_type_intrinsic) {
               const nir_intrinsic_op op =
                  nir_instr_as_intrinsic(instr)->intrinsic;
               bindless_handles += op == nir_intrinsic_bindless_image_agx ||
                                   op == nir_intrinsic_bindless_sampler_agx;
               bindless_stores += op == nir_intrinsic_bindless_image_store;
               ordinary_stores += op == nir_intrinsic_image_store;
            }
         }
      }

      const bool ok = mode == 0
         ? lowered && binding.texture_set == 0 && binding.pbe_set == 1 &&
              binding.texture_offset_B == 0 &&
              binding.sampler_index_offset_B == 40 &&
              binding.pbe_offset_B == 32 && bindless_handles == 0 &&
              bindless_stores == 0 && ordinary_stores == 1 && samples == 1
         : !lowered && binding.texture_offset_B == UINT32_MAX &&
              binding.sampler_index_offset_B == UINT32_MAX &&
              binding.pbe_offset_B == UINT32_MAX &&
              binding.texture_set == UINT8_MAX &&
              binding.pbe_set == UINT8_MAX && bindless_handles == 3 &&
              bindless_stores == 1 && ordinary_stores == 0 && samples == 1;
      ralloc_free(nir);
      if (!ok)
         return false;
   }
   return true;
}

static bool
test_indexed_compute_sample(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "AGX3 indexed compute sample");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *element = nir_channel(&b, gid, 0);
   nir_def *offset = nir_imul_imm(&b, element, 72);
   nir_def *set = nir_load_descriptor_set_agx(&b, 4);
   nir_def *entry = nir_iadd(&b, set, nir_u2u64(&b, offset));
   nir_def *sampler_index = nir_load_global_constant(
      &b, 1, 16, nir_iadd_imm(&b, entry, 40), .align_mul = 2);
   nir_def *image = nir_bindless_image_agx(&b, offset, .desc_set = 4);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_iadd_imm(&b, offset, 32), sampler_index, .desc_set = 4);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_vec2(&b, nir_u2f32(&b, element), nir_imm_float(&b, 0.5f)));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   tex->src[3] =
      nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 0.0f));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_export_agx(&b, nir_channel(&b, &tex->def, 0), .base = 0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   bool ok = agx_nir_lower_g17p_indexed_samplers(b.shader);
   const int ti = nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
   const int si = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   ok &= ti >= 0 && si >= 0 &&
         nir_src_bit_size(tex->src[ti].src) == 64 &&
         nir_src_bit_size(tex->src[si].src) == 64;

   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
            lower_test_descriptor_set, nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   key.reserved_preamble = 32;
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &key, &part) && part.binary != NULL;
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static bool
test_indexed_compute_storage(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "AGX3 indexed compute sample and formatted store");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *element = nir_channel(&b, gid, 0);
   nir_def *offset = nir_imul_imm(&b, element, 72);
   nir_def *set = nir_load_descriptor_set_agx(&b, 4);
   nir_def *entry = nir_iadd(&b, set, nir_u2u64(&b, offset));
   nir_def *sampler_index = nir_load_global_constant(
      &b, 1, 16, nir_iadd_imm(&b, entry, 40), .align_mul = 2);
   nir_def *image = nir_bindless_image_agx(&b, offset, .desc_set = 4);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_iadd_imm(&b, offset, 32), sampler_index, .desc_set = 4);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_vec2(&b, nir_u2f32(&b, element), nir_imm_float(&b, 0.5f)));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   tex->src[3] =
      nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 0.0f));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *coord = nir_vec4(&b, element, zero, zero, zero);
   nir_def *pbe = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 176), .desc_set = 0);
   nir_def *data = nir_fadd_imm(&b, &tex->def, 0.25f);
   nir_bindless_image_store(
      &b, pbe, coord, zero, data, zero, .image_dim = GLSL_SAMPLER_DIM_2D,
      .image_array = false, .format = PIPE_FORMAT_R10G10B10A2_UNORM,
      .access = ACCESS_RESTRICT);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   struct agx_g17p_indexed_storage_binding binding = {
      .pbe_offset_B = UINT32_MAX,
      .pbe_set = UINT8_MAX,
   };
   bool ok = agx_nir_lower_g17p_indexed_samplers(b.shader) &&
             agx_nir_lower_g17p_indexed_storage(b.shader, &binding) &&
             binding.pbe_set == 0 && binding.pbe_offset_B == 176;

   unsigned bindless_stores = 0, ordinary_stores = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         const nir_intrinsic_op op =
            nir_instr_as_intrinsic(instr)->intrinsic;
         bindless_stores += op == nir_intrinsic_bindless_image_store;
         ordinary_stores += op == nir_intrinsic_image_store;
      }
   }
   ok &= bindless_stores == 0 && ordinary_stores == 1;

   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
            lower_test_descriptor_set, nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   key.reserved_preamble = 80;
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &key, &part) && part.binary != NULL;
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static bool
test_g17p_eot_block_store(void)
{
   /* Genuine applegpu-nt builtin EOT, key0x10020300, RTmask1. Native fragment
    * metadata requests tableVA64 at u0 and imageblock byteoffset32 at u4;
    * common0=3 GPRdwords, fields38/42 D=16. Not a compute-wrapper prefix. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P RT0 EOT");
   nir_def *coords =
      nir_trim_vector(&b, nir_load_global_invocation_id(&b, 32), 2);
   nir_image_store_block_agx(
      &b, nir_imm_intN_t(&b, 0, 16), nir_imm_int(&b, 0), coords,
      .format = PIPE_FORMAT_R8_UNORM, .image_dim = GLSL_SAMPLER_DIM_2D,
      .explicit_coord = true);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   static const uint8_t expected[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x75, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   bool ok = compiled && part.info.binary_size == sizeof(expected) &&
             part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
             part.info.agx3_staging_halfs == 8 &&
             memcmp(part.binary, expected, sizeof(expected)) == 0;

   ok &= sizeof(expected) == 68 && part.info.main_offset == 0 &&
         part.info.main_size == 68;
   if (!ok) {
      fprintf(stderr,
              "AGX3b EOT block-store compiler mismatch: compiled=%u size=%u "
              "gprs=%u\n",
              compiled, part.info.binary_size, part.info.nr_gprs);
      if (compiled) {
         for (unsigned i = 0; i < part.info.binary_size; ++i)
            fprintf(stderr, "%02x", ((const uint8_t *)part.binary)[i]);
         fputc('\n', stderr);
      }
   }

   free(part.binary);
   ralloc_free(b.shader);

   /* Exact Quake WBOIT tile layout: native builtin EOT mask3 stores an
    * RGBA16F block at byte0 and scalar R8 at byte8. */
   nir_builder mrt = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P WBOIT MRT EOT");
   coords = nir_trim_vector(&mrt, nir_load_global_invocation_id(&mrt, 32), 2);
   nir_image_store_block_agx(
      &mrt, nir_imm_intN_t(&mrt, 0, 16), nir_imm_int(&mrt, 0), coords,
      .format = PIPE_FORMAT_R16_FLOAT,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   nir_image_store_block_agx(
      &mrt, nir_imm_intN_t(&mrt, 1, 16), nir_imm_int(&mrt, 8), coords,
      .format = PIPE_FORMAT_R8_UNORM, .image_dim = GLSL_SAMPLER_DIM_2D,
      .explicit_coord = true);
   agx_preprocess_nir(mrt.shader);
   memset(&part, 0, sizeof(part));
   const bool mrt_compiled = agx_compile_shader_nir(mrt.shader, &key, &part);
   static const uint8_t expected_mrt[] = {
      0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,0x0c,0x9d,0x10,0x26,
      0x24,0xa9,0x10,0x26,0x9f,0x10,0x54,0x02,0x03,0x04,0x14,0x00,
      0x50,0x26,0x0a,0x00,0x9f,0x20,0x54,0x00,0x03,0x00,0x10,0x00,
      0x50,0x26,0x0a,0x00,0x97,0x05,0x54,0x04,0x00,0x08,0x00,0x00,
      0xd0,0x40,0x10,0x00,0x0b,0x82,0x21,0x04,0x1b,0x80,0x21,0x04,
      0x97,0x05,0x54,0x0c,0x00,0x10,0x00,0x00,0xd0,0x44,0x10,0x00,
      0x4b,0x03,0x2f,0x00,0x42,0x00,0x00,0x04,0x00,0x00,0x5b,0x01,
      0x2f,0x00,0x42,0x00,0x00,0x04,0x00,0x00,0x57,0x10,0x54,0x00,
      0x00,0x00,0x00,0x80,0xa8,0x15,0x00,0x00,0x57,0x20,0x54,0x08,
      0x00,0x10,0x00,0x90,0xa8,0x75,0x00,0x00,0x0e,0x00,0x00,0x00,
   };
   ok &= mrt_compiled && part.info.binary_size == sizeof(expected_mrt) &&
         part.info.main_offset == 0 && part.info.main_size == sizeof(expected_mrt) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 16 &&
         memcmp(part.binary, expected_mrt, sizeof(expected_mrt)) == 0;
   if (mrt_compiled)
      free(part.binary);
   ralloc_free(mrt.shader);

   /* Genuine applegpu-nt builtin EOT key0x04021100: scalar I32 tile store. */
   nir_builder i32 = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P I32 RT0 EOT");
   coords =
      nir_trim_vector(&i32, nir_load_global_invocation_id(&i32, 32), 2);
   nir_image_store_block_agx(
      &i32, nir_imm_intN_t(&i32, 0, 16), nir_imm_int(&i32, 0), coords,
      .format = PIPE_FORMAT_R32_UINT, .image_dim = GLSL_SAMPLER_DIM_2D,
      .explicit_coord = true);
   agx_preprocess_nir(i32.shader);
   memset(&part, 0, sizeof(part));
   key.g17p_rt0_eot_i32 = true;
   key.g17p_rt0_eot_words = 1;
   const bool i32_compiled = agx_compile_shader_nir(i32.shader, &key, &part);
   static const uint8_t expected_i32[] = {
      0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,0x0c,0x9d,0x10,0x26,
      0x24,0xa9,0x10,0x26,0x9f,0x10,0x54,0x00,0x00,0x04,0x14,0x00,
      0x50,0x06,0x0b,0x00,0x9f,0x20,0x54,0x02,0x00,0x00,0x10,0x00,
      0x50,0x06,0x0b,0x00,0x97,0x05,0x54,0x04,0x00,0x08,0x00,0x00,
      0xd0,0x40,0x10,0x00,0x57,0x10,0x54,0x00,0x00,0x00,0x00,0x80,
      0xa8,0x35,0x00,0x00,0x0e,0x00,0x00,0x00,
   };
   ok &= i32_compiled && part.info.binary_size == sizeof(expected_i32) &&
         part.info.main_offset == 0 &&
         part.info.main_size == sizeof(expected_i32) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 &&
         memcmp(part.binary, expected_i32, sizeof(expected_i32)) == 0;
   if (i32_compiled)
      free(part.binary);
   ralloc_free(i32.shader);
   key.g17p_rt0_eot_i32 = false;
   key.g17p_rt0_eot_words = 0;

   /* Fresh Apple builtin key0x10020900: an RGBA16_UINT attachment lowers to
    * exactly two I16 imageblock words and selects the native 0x15 EOT. */
   nir_builder i16 = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P RGBA16 UINT RT0 EOT");
   coords =
      nir_trim_vector(&i16, nir_load_global_invocation_id(&i16, 32), 2);
   nir_image_store_block_agx(
      &i16, nir_imm_intN_t(&i16, 0, 16), nir_imm_int(&i16, 0), coords,
      .format = PIPE_FORMAT_R16_UINT, .image_dim = GLSL_SAMPLER_DIM_2D,
      .explicit_coord = true);
   agx_preprocess_nir(i16.shader);
   memset(&part, 0, sizeof(part));
   key.g17p_rt0_eot_words = 2;
   const bool i16_compiled = agx_compile_shader_nir(i16.shader, &key, &part);
   static const uint8_t expected_i16[] = {
      0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,0x0c,0x9d,0x10,0x26,
      0x24,0xa9,0x10,0x26,0x9f,0x10,0x54,0x00,0x00,0x04,0x14,0x00,
      0x50,0x06,0x0b,0x00,0x9f,0x20,0x54,0x02,0x00,0x00,0x10,0x00,
      0x50,0x06,0x0b,0x00,0x97,0x05,0x54,0x04,0x00,0x08,0x00,0x00,
      0xd0,0x40,0x10,0x00,0x57,0x10,0x54,0x00,0x00,0x00,0x00,0x80,
      0xa8,0x15,0x00,0x00,0x0e,0x00,0x00,0x00,
   };
   ok &= i16_compiled && part.info.binary_size == sizeof(expected_i16) &&
         part.info.main_offset == 0 &&
         part.info.main_size == sizeof(expected_i16) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 &&
         memcmp(part.binary, expected_i16, sizeof(expected_i16)) == 0;
   if (i16_compiled)
      free(part.binary);
   ralloc_free(i16.shader);

   /* Do not let an ungrounded I16 word count inherit the same selector. */
   nir_builder bad_i16 = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "unsupported G17P I16 EOT");
   coords = nir_trim_vector(
      &bad_i16, nir_load_global_invocation_id(&bad_i16, 32), 2);
   nir_image_store_block_agx(
      &bad_i16, nir_imm_intN_t(&bad_i16, 0, 16), nir_imm_int(&bad_i16, 0),
      coords, .format = PIPE_FORMAT_R16_UINT,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   agx_preprocess_nir(bad_i16.shader);
   memset(&part, 0xa5, sizeof(part));
   key.g17p_rt0_eot_words = 3;
   const bool bad_i16_compiled =
      agx_compile_shader_nir(bad_i16.shader, &key, &part);
   const struct agx_shader_part cleared_i16 = {0};
   if (bad_i16_compiled ||
       memcmp(&part, &cleared_i16, sizeof(part)) != 0) {
      fprintf(stderr, "unsupported G17P I16 word count did not fail closed\n");
      ok = false;
   }
   if (bad_i16_compiled)
      free(part.binary);
   ralloc_free(bad_i16.shader);
   key.g17p_rt0_eot_words = 0;

   /* A different store format must not inherit RT0's byte string. */
   nir_builder bad = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "unsupported G17P EOT");
   coords = nir_trim_vector(&bad, nir_load_global_invocation_id(&bad, 32), 2);
   nir_image_store_block_agx(
      &bad, nir_imm_intN_t(&bad, 0, 16), nir_imm_int(&bad, 0), coords,
      .format = PIPE_FORMAT_R16_FLOAT, .image_dim = GLSL_SAMPLER_DIM_2D,
      .explicit_coord = true);
   agx_preprocess_nir(bad.shader);
   memset(&part, 0xa5, sizeof(part));
   const bool bad_compiled = agx_compile_shader_nir(bad.shader, &key, &part);
   const bool rejected = !bad_compiled;
   const struct agx_shader_part cleared = {0};
   if (!rejected || memcmp(&part, &cleared, sizeof(part)) != 0) {
      fprintf(stderr, "unsupported G17P EOT format did not fail closed\n");
      ok = false;
   }
   if (bad_compiled)
      free(part.binary);
   ralloc_free(bad.shader);
   return ok;
}

/*
 * Fragment shader sampling texture 0 through sampler 0 with implicit LOD.
 * Keep both index ABIs explicit: ordinary bound sampling reserves sampler 0
 * for texel fetch, while G17P's compact pointer tables select native slot 0.
 * The compact reference is constructed directly at the post-lowering level;
 * it must not run the combined-sampler pass that it independently checks.
 */
static nir_shader *
build_fs_texture_index_abi(bool compact_native)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 textured-triangle FS");

   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *uv = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   tex->src[0] = nir_tex_src_for_ssa(compact_native ? nir_tex_src_backend1
                                                 : nir_tex_src_coord,
                                     uv);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   fs_store_colour(&b, &tex->def);

   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_early, false);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   return b.shader;
}

static nir_shader *
build_fs_texture_imm(void)
{
   return build_fs_texture_index_abi(false);
}

static nir_shader *
build_fs_texture_compact_native(void)
{
   return build_fs_texture_index_abi(true);
}

static bool
has_compact_native_sample(const nir_shader *shader)
{
   unsigned samples = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               const nir_intrinsic_op op =
                  nir_instr_as_intrinsic(instr)->intrinsic;
               if (op == nir_intrinsic_bindless_image_agx ||
                   op == nir_intrinsic_bindless_sampler_agx ||
                   op == nir_intrinsic_load_descriptor_set_agx ||
                   op == nir_intrinsic_load_root_agx)
                  return false;
            }

            if (instr->type != nir_instr_type_tex)
               continue;

            const nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (++samples != 1 || tex->op != nir_texop_tex ||
                tex->texture_index != 0 || tex->sampler_index != 0 ||
                tex->num_srcs != 1 ||
                tex->src[0].src_type != nir_tex_src_backend1 ||
                tex->sampler_dim != GLSL_SAMPLER_DIM_2D || tex->is_array ||
                tex->is_shadow || tex->is_sparse || tex->backend_flags != 0 ||
                tex->coord_components != 2 ||
                nir_src_num_components(tex->src[0].src) != 2 ||
                nir_src_bit_size(tex->src[0].src) != 32 ||
                tex->dest_type != nir_type_float32 ||
                tex->def.num_components != 4 || tex->def.bit_size != 32)
               return false;
         }
      }
   }
   return samples == 1;
}

static bool
test_texture_index_abis(void)
{
   nir_shader *legacy = build_fs_texture_imm();
   nir_shader *native = build_fs_texture_compact_native();
   unsigned samples = 0;
   bool legacy_ok = true;

   nir_foreach_function_impl(impl, legacy) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            const nir_tex_instr *tex = nir_instr_as_tex(instr);
            const int sampler =
               nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
            ++samples;
            legacy_ok &= sampler >= 0 &&
                         nir_src_is_const(tex->src[sampler].src) &&
                         nir_src_as_uint(tex->src[sampler].src) == 1;
         }
      }
   }

   bool ok = legacy_ok && samples == 1 &&
             !has_compact_native_sample(legacy) &&
             has_compact_native_sample(native);

   /* The old specialized result had no handles, but still selected sampler
    * 1. Reject that exact regression, independently of the source count. */
   nir_foreach_function_impl(impl, native) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               nir_instr_as_tex(instr)->sampler_index = 1;
         }
      }
   }
   ok &= !has_compact_native_sample(native);
   ralloc_free(legacy);
   ralloc_free(native);
   return ok;
}

/* Descriptor set 0 is placed in fragment uniform halfs 16..19. Load one
 * vec4 directly from that descriptor resource and write it to RT0. */
static nir_shader *
build_fs_descriptor_resource(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 descriptor resource FS");

   nir_def *set = nir_load_preamble(&b, 1, 64, .base = 16);
   nir_def *raw = nir_load_agx(&b, 4, 32, set, nir_imm_int(&b, 0),
                               .access = ACCESS_CAN_SPECULATE, .base = 0,
                               .format = PIPE_FORMAT_R32_UINT,
                               .sign_extend = false);
   fs_store_colour(&b, nir_u2f32(&b, raw));
   return fs_finish(&b);
}

/* The bounded buffer-to-image meta fragment shape: use the root uniform to
 * fetch a staging-buffer address, index it by pixel coordinate, unpack one
 * RGBA8 texel, and write RT0.
 */
static nir_shader *
build_fs_meta_rgba_upload(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 RGBA upload FS");

   nir_def *root =
      nir_load_preamble(&b, 1, 64, .base = AGX_ABI_FUNI_ROOT);
   nir_def *source = nir_load_global_constant(
      &b, 1, 64, root, .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
   nir_def *xy = nir_u2u32(&b, nir_load_pixel_coord(&b));
   nir_def *pixel = nir_iadd(
      &b, nir_channel(&b, xy, 0),
      nir_imul_imm(&b, nir_channel(&b, xy, 1), 256));
   nir_def *address =
      nir_iadd(&b, source, nir_u2u64(&b, nir_imul_imm(&b, pixel, 4)));
   nir_def *packed =
      nir_load_global(&b, 1, 32, address, .align_mul = 4);
   nir_def *channel[4];
   for (unsigned i = 0; i < ARRAY_SIZE(channel); ++i) {
      const unsigned shift = i * 8;
      nir_def *masked =
         nir_iand(&b, packed, nir_imm_int(&b, 0xffu << shift));
      channel[i] = nir_fmul_imm(
         &b, nir_u2f32(&b, masked),
         1.0f / (255.0f * (float)(1u << shift)));
   }
   nir_def *colour =
      nir_vec4(&b, channel[0], channel[1], channel[2], channel[3]);

   fs_store_colour(&b, colour);
   return fs_finish(&b);
}

/*
 * Fragment shader exercising the funop family: floor/exp2/log2 (f32rint and
 * friends) plus the fragment-only gradient dfdx.
 */
static nir_shader *
build_fs_funops(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 funop FS");

   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *v = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);

   nir_def *x = nir_channel(&b, v, 0);
   nir_def *y = nir_channel(&b, v, 1);

   fs_store_colour(&b, nir_vec4(&b, nir_ffloor(&b, x), nir_fexp2(&b, y),
                                nir_flog2(&b, nir_fabs(&b, x)),
                                nir_ddx(&b, y)));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

struct agx3_graphics_case {
   const char *name;
   nir_shader *(*build)(void);
   bool expect_ok;
   bool lower_combined_sampler;
   uint16_t reserved_preamble;
   uint16_t minimum_binary_size;
   uint8_t expected_leading_sr;
   const uint8_t *expected_g17p_binary;
   size_t expected_g17p_binary_size;
   bool compare_apple_g17p_vs;
   uint8_t expected_g17p_loads;
   bool g17p_only;
};

static const struct agx3_graphics_case cases[] = {
   {"vs-flat-triangle", build_vs_flat, true, .expected_leading_sr = 0xdd},
   {"vs-g17p-div3-mul-high", build_vs_g17p_div3_mul_high, true,
    .expected_leading_sr = 0xdd, .g17p_only = true},
   {"vs-instance-id", build_vs_instance_id, true,
    .expected_leading_sr = 0xd8},
   {"vs-attribute-fetch", build_vs_attribute, true,
    .expected_leading_sr = 0xdd},
   {"vs-uniform-attribute-fetch", build_vs_uniform_attribute, true},
   {"vs-g17p-float2-position", build_vs_g17p_float2_position, true,
    .expected_g17p_binary = mesa_g17p_float2_vs,
    .expected_g17p_binary_size = sizeof(mesa_g17p_float2_vs),
    .compare_apple_g17p_vs = true, .expected_g17p_loads = 1},
   {"vs-g17p-single-instanced-varying", build_vs_g17p_single_instanced_varying,
    true, .expected_g17p_loads = 1},
   {"vs-g17p-two-loads", build_vs_g17p_two_loads, true,
    .expected_g17p_loads = 2},
   {"vs-g17p-half2-varying", build_vs_g17p_half2_varying, true,
    .expected_g17p_binary = mesa_g17p_half2_varying_vs,
    .expected_g17p_binary_size = sizeof(mesa_g17p_half2_varying_vs),
    .expected_g17p_loads = 2, .g17p_only = true},
   {"vs-vkcube-varyings", build_vs_vkcube_varyings, true},
   {"vs-vkcube-vbo-ubo", build_vs_vkcube_vbo_ubo, true, false,
    AGX_ABI_VUNI_COUNT_VK(2) + 8, 257},
   {"fs-flat-colour", build_fs_flat, true,
    .expected_g17p_binary = g17p_flat_colour,
    .expected_g17p_binary_size = sizeof(g17p_flat_colour)},
   {"fs-interpolated-varying", build_fs_interpolated, true},
   {"fs-tilebuffer-load", build_fs_tilebuffer_load, true},
   {"fs-textured-triangle", build_fs_texture_imm, true},
   {"fs-vkcube-combined-sampler", build_fs_texture, true, true},
   {"fs-descriptor-resource", build_fs_descriptor_resource, true},
   {"fs-meta-rgba-upload", build_fs_meta_rgba_upload, true, false,
    AGX_ABI_FUNI_ROOT + 4},
   {"fs-transcendentals", build_fs_funops, true},

   /*
    * Deliberate refusals. Each one names a specific missing witness, not a
    * missing line of routing:
    *
    * - discard lowers to sample_mask. Apple's discard and [[sample_mask]]
    *   shaders both go through movmsk under the class-1 pixwait/pixrel
    *   bracket with a value staged by a copy (never cmsk, which only
    *   alpha-to-coverage emits), and movmsk's operand roles plus the staging
    *   ABI are not isolated. Mesa's mask+value operand split has no
    *   witnessed encoding either way.
    *
    * - a depth write lowers to zs_emit, whose counterpart is depthf. depthf's
    *   enable bits are grounded, but it carries no value register: every
    *   witness stages the value through a copy whose destination field is 0,
    *   behind an un-decomposed iadd/fsel cluster that may be a required
    *   format transform, so the staging ABI is a correlation rather than a
    *   traced contract (see agx3_pack_binary.c).
    *
    */
   {"fs-discard", build_fs_discard, false},
   {"fs-depth-write", build_fs_depth, false},
};

static bool
run_case(const struct agx3_graphics_case *c, enum agx_isa_encoding enc)
{
   nir_shader *nir = c->build();

   if (c->lower_combined_sampler) {
      struct agx_g17p_combined_sampler_binding binding;
      if (!agx_nir_lower_g17p_combined_sampler(nir, &binding) ||
          binding.texture_offset_B != 64 ||
          binding.sampler_index_offset_B != 104 ||
          !has_compact_native_sample(nir)) {
         fprintf(stderr, "%s %s: combined descriptor lowering mismatch\n",
                 enc_name(enc), c->name);
         ralloc_free(nir);
         return false;
      }
   }

   struct agx_shader_key key = {0};
   key.dev.encoding = enc;
   key.no_preamble = true;
   key.reserved_preamble = c->reserved_preamble;

   struct agx_shader_part part;
   memset(&part, 0xa5, sizeof(part));

   const bool ok = agx_compile_shader_nir(nir, &key, &part);
   bool result;

   if (!ok) {
      const uint8_t zero[sizeof(part)] = {0};
      if (memcmp(&part, zero, sizeof(part)) != 0) {
         fprintf(stderr, "%s %s: rejection did not clear compiler output\n",
                 enc_name(enc), c->name);
         ralloc_free(nir);
         return false;
      }

      result = !c->expect_ok;
      if (!result)
         fprintf(stderr, "%s %s: REJECTED by agx3_pack_binary\n", enc_name(enc),
                 c->name);
   } else {
      result = c->expect_ok && part.binary != NULL &&
               part.info.binary_size > sizeof(agx3_end) &&
               part.info.main_offset == 0 &&
               memcmp((const uint8_t *)part.binary + part.info.binary_size -
                         sizeof(agx3_end),
                      agx3_end, sizeof(agx3_end)) == 0;

      if (result && c->reserved_preamble != 0) {
         result = !part.info.has_preamble &&
                  part.info.push_count == c->reserved_preamble;
      }

      if (result && c->minimum_binary_size != 0)
         result = part.info.binary_size >= c->minimum_binary_size;

      if (result && c->expected_leading_sr != 0) {
         const uint8_t expected[] = {
            0x0c, c->expected_leading_sr, 0x10, 0x06,
         };
         result = part.info.binary_size >= sizeof(expected) &&
                  memcmp(part.binary, expected, sizeof(expected)) == 0;
         if (!result) {
            fprintf(stderr, "%s %s: missing leading SR %#x rld\n",
                    enc_name(enc), c->name, c->expected_leading_sr);
         }
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->expected_g17p_binary != NULL) {
         const bool resources_unchanged =
            !part.info.has_preamble && part.info.push_count == 0 &&
            part.info.scratch_size == 0 && !part.info.uses_txf &&
            !part.info.uses_sampler_heap &&
            part.info.texture_state_count == 0 &&
            part.info.sampler_state_count == 0;
         result = resources_unchanged &&
                  part.info.binary_size == c->expected_g17p_binary_size &&
                  memcmp(part.binary, c->expected_g17p_binary,
                         c->expected_g17p_binary_size) == 0;
         if (!result) {
            fprintf(stderr,
                    "%s %s: G17P binary/resource metadata mismatch\n",
                    enc_name(enc), c->name);
            fprintf(stderr, "actual main (%u bytes, staging %u halfs): ",
                    part.info.binary_size, part.info.agx3_staging_halfs);
            for (unsigned j = 0; j < part.info.binary_size; ++j)
               fprintf(stderr, "%02x", ((const uint8_t *)part.binary)[j]);
            fprintf(stderr, "\n");
         }
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->compare_apple_g17p_vs) {
         result = part.info.agx3_staging_known && part.info.agx3_staging_halfs == 8 &&
                  g17p_vs_matches_apple(part.binary, part.info.binary_size);
         if (!result) {
            fprintf(stderr, "%s %s: Apple-88 VS protocol mismatch\n",
                    enc_name(enc), c->name);
         }
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->expected_g17p_loads != 0) {
         result = g17p_vertex_load_policy(part.binary, part.info.binary_size,
                                           c->expected_g17p_loads);
         if (!result) {
            fprintf(stderr, "%s %s: vertex LDDEV policy mismatch\n",
                    enc_name(enc), c->name);
         }
      }

      if (result && c->lower_combined_sampler) {
         nir_shader *reference_nir = build_fs_texture_compact_native();
         struct agx_shader_part reference = {0};
         const bool reference_ok =
            has_compact_native_sample(reference_nir) &&
            agx_compile_shader_nir(reference_nir, &key, &reference);
         const bool binary_matches =
            reference_ok &&
            reference.info.binary_size == part.info.binary_size &&
            memcmp(reference.binary, part.binary, part.info.binary_size) == 0;
         result = binary_matches && !part.info.has_preamble &&
                  !part.info.uses_sampler_heap &&
                  part.info.texture_state_count == 0 &&
                  part.info.sampler_state_count == 0;
         if (!result) {
            fprintf(stderr,
                    "%s %s: specialized=%u reference=%u binary_match=%u "
                    "preamble=%u heap=%u ts=%u ss=%u\n",
                    enc_name(enc), c->name, part.info.binary_size,
                    reference.info.binary_size, binary_matches,
                    part.info.has_preamble, part.info.uses_sampler_heap,
                    part.info.texture_state_count,
                    part.info.sampler_state_count);
         }
         free(reference.binary);
         ralloc_free(reference_nir);
      }

      if (!result) {
         fprintf(stderr, "%s %s: %s\n", enc_name(enc), c->name,
                 c->expect_ok ? "accepted but binary is malformed"
                              : "unexpectedly accepted");
      } else if (getenv("AGX3_DUMP_BINARY")) {
         fprintf(stderr, "%s %s: %u bytes:", enc_name(enc), c->name,
                 part.info.binary_size);
         for (unsigned i = 0; i < part.info.binary_size; ++i)
            fprintf(stderr, " %02x", ((const uint8_t *)part.binary)[i]);
         fprintf(stderr, "\n");
      }

      free(part.binary);
   }

   ralloc_free(nir);
   return result;
}

static bool
test_g17p_half2_varying(void)
{
   const struct agx3_graphics_case c = {
      "vs-g17p-half2-varying", build_vs_g17p_half2_varying, true,
      .expected_g17p_binary = mesa_g17p_half2_varying_vs,
      .expected_g17p_binary_size = sizeof(mesa_g17p_half2_varying_vs),
      .expected_g17p_loads = 2,
   };
   return run_case(&c, AGX_ISA_ENCODING_AGX3B);
}

int
main(int argc, char **argv)
{
   if (argc == 2 && !strcmp(argv[1], "--half-vbo-only")) {
      const bool ok = test_g17p_half2_varying();
      fprintf(stderr, "G17P R16G16_SFLOAT vertex attribute: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--eot-only")) {
      const bool ok = test_g17p_eot_block_store();
      printf("G17P RT0 and WBOIT MRT EOT: %s\n", ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--custom-border-only")) {
      bool ok = test_custom_border_samplers(false, false) &&
                test_custom_border_samplers(true, false) &&
                test_custom_border_samplers(true, 1) &&
                test_custom_border_samplers(true, 2) &&
                test_indexed_sampler_sets() && test_indexed_fetch_sets();
      printf("custom border runtime branch, exact indices, mixed reads: %s\n",
             ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--indexed-fetch-only")) {
      const bool ok=test_indexed_fetch_sets();
      printf("mixed indexed texture reads, sample, sets and arrays: %s\n",ok?"PASS":"FAIL");
      return ok?0:1;
   }
   if (argc == 2 && !strcmp(argv[1], "--indexed-samplers-only")) {
      const bool ok = test_indexed_sampler_sets() &&
                      test_indexed_compute_sample();
      printf("indexed graphics/compute sampler sets and live roots: %s\n",
             ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--indexed-storage-only")) {
      const bool ok = test_indexed_compute_storage();
      printf("indexed compute samples plus RGB10 storage: %s\n",
             ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--sampler-root-only")) {
      const bool ok = test_combined_sampler_root() && test_combined_sampler_scope();
      fprintf(stderr, "native sampler plus push-root lowering: %s\n", ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--reload-staging-only")) {
      const bool ok = test_g17p_background_reload_staging();
      fprintf(stderr, "G17P background reload staging: %s\n", ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--f16-clear-only")) {
      const bool ok = test_g17p_background_f16_clear_narrowing();
      fprintf(stderr, "G17P f16 background clear narrowing: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--storage-image-lowering-only")) {
      const bool ok = test_storage_image_lowering();
      fprintf(stderr, "G17P bounded RGBA8UI storage-image lowering: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--sampled-storage-lowering-only")) {
      const bool ok = test_sampled_storage_lowering();
      fprintf(stderr, "G17P sampled RGBA8 storage-image lowering: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--cube-sample-lowering-only")) {
      const bool ok = test_g17p_cube_sample_lowering();
      fprintf(stderr, "G17P cube-to-array sample lowering: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--2d-robustness-only")) {
      const bool ok = test_g17p_2d_robustness_scope();
      fprintf(stderr, "G17P 2D image robustness scope: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   const enum agx_isa_encoding encodings[] = {
      AGX_ISA_ENCODING_AGX3B,
      AGX_ISA_ENCODING_AGX3C,
   };
   bool all_ok = test_custom_border_samplers(false, 0) &&
                 test_custom_border_samplers(true, 0) &&
                 test_custom_border_samplers(true, 1) &&
                 test_custom_border_samplers(true, 2);

   if (!test_texture_index_abis()) {
      fputs("legacy and compact-native sampler index ABIs conflated\n", stderr);
      all_ok = false;
   }

   if (!test_combined_sampler_scope() || !test_combined_sampler_root()) {
      fputs("combined sampler lowering consumed a non-sampler bias\n", stderr);
      all_ok = false;
   }

   if (!test_txf_lowering()) {
      fputs("static TXF descriptor lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_indexed_compute_sample()) {
      fputs("indexed compute sampler lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_indexed_compute_storage()) {
      fputs("indexed compute storage lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_storage_image_lowering()) {
      fputs("bounded storage-image lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_sampled_storage_lowering()) {
      fputs("sampled storage-image lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_g17p_cube_sample_lowering()) {
      fputs("cube-to-array sample lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_g17p_2d_robustness_scope()) {
      fputs("2D image robustness scope mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_g17p_eot_block_store())
      all_ok = false;

   if (!test_g17p_background_reload_staging())
      return 1;
   if (!test_g17p_background_clear())
      all_ok = false;
   if (!test_g17p_background_f16_clear_narrowing())
      all_ok = false;

   for (unsigned e = 0; e < ARRAY_SIZE(encodings); ++e) {
      for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
         if (cases[i].g17p_only && encodings[e] != AGX_ISA_ENCODING_AGX3B)
            continue;
         if (!run_case(&cases[i], encodings[e]))
            all_ok = false;
      }
   }

   if (!all_ok)
      return 1;

   puts("agx3_graphics: vertex and fragment shaders compile");
   return 0;
}
