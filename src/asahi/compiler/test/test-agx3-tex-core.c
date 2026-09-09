/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_tex_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static bool
same(const struct agx3_tex_fetch_desc *a,
     const struct agx3_tex_fetch_desc *b)
{
   return a->dest == b->dest && a->coord == b->coord &&
          a->width == b->width && a->mask == b->mask &&
          a->texture == b->texture && a->slot == b->slot &&
          a->wait_mask == b->wait_mask && a->coord_kill == b->coord_kill &&
          a->final == b->final &&
          a->array == b->array &&
          a->multisample == b->multisample && a->sample == b->sample &&
          (a->texture_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) ==
             (b->texture_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) &&
          (a->sampler_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) ==
             (b->sampler_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) &&
          a->texture_value == b->texture_value &&
          a->sampler_value == b->sampler_value &&
          a->texture_kill == b->texture_kill;
}

static int
roundtrip(enum agx_isa_encoding enc, const struct agx3_tex_fetch_desc *d)
{
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length = 0;
   struct agx3_tex_fetch_desc decoded;
   CHECK(agx3_tex_encode_fetch(enc, d, bytes, &length));
   CHECK(length == (enc == AGX_ISA_ENCODING_AGX3B ?
      ((d->texture_kind <= AGX3_TEX_OPERAND_IMMEDIATE &&
        d->sampler_kind <= AGX3_TEX_OPERAND_IMMEDIATE) ? 14 : 20) : 8));
   CHECK(agx3_tex_decode_fetch(enc, bytes, length, &decoded));
   CHECK(same(d, &decoded));
   return 0;
}

static int
test_exact_witnesses(void)
{
   static const uint8_t b[] = {
      0x05, 0x80, 0x0c, 0xb8, 0xb0, 0x00, 0x17,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
   };
   static const uint8_t c[] = {0x0f, 0x06, 0x1a, 0x00,
                               0x40, 0x80, 0x60, 0x29};
   static const uint8_t b_mask1_d0[] = {
      0x05, 0x80, 0x0c, 0xa0, 0x90, 0x00, 0x17,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
   };
   static const uint8_t b_mask1_d2[] = {
      0x0d, 0x80, 0x0c, 0xa0, 0x90, 0x00, 0x17,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
   };
   static const uint8_t c_mask1_d0[] = {0x0f, 0x06, 0x1a, 0x00,
                                        0x40, 0x80, 0x00, 0x21};
   static const uint8_t c_mask1_d2[] = {0x1f, 0x06, 0x1a, 0x00,
                                        0x40, 0x80, 0x00, 0x21};
   struct agx3_tex_fetch_desc d = {
      .dest = 0,
      .coord = 0,
      .width = AGX3_TEX_W32,
      .mask = 15,
      .texture = 0,
      .slot = 1,
      .wait_mask = 1,
      .coord_kill = true,
      .final = true,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length;
   struct agx3_tex_fetch_desc decoded;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == sizeof(b) && memcmp(bytes, b, sizeof(b)) == 0);
   CHECK(agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, b, sizeof(b), &decoded));
   CHECK(same(&d, &decoded));
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   CHECK(length == sizeof(c) && memcmp(bytes, c, sizeof(c)) == 0);
   CHECK(agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3C, c, sizeof(c), &decoded));
   CHECK(same(&d, &decoded));

   d.mask = 1;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == sizeof(b_mask1_d0) &&
         memcmp(bytes, b_mask1_d0, sizeof(b_mask1_d0)) == 0);
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   CHECK(length == sizeof(c_mask1_d0) &&
         memcmp(bytes, c_mask1_d0, sizeof(c_mask1_d0)) == 0);
   d.dest = 2;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == sizeof(b_mask1_d2) &&
         memcmp(bytes, b_mask1_d2, sizeof(b_mask1_d2)) == 0);
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   CHECK(length == sizeof(c_mask1_d2) &&
         memcmp(bytes, c_mask1_d2, sizeof(c_mask1_d2)) == 0);
   return 0;
}

static int
test_supported_axes(void)
{
   for (unsigned ei = 0; ei < 2; ++ei) {
      const enum agx_isa_encoding enc =
         ei ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      struct agx3_tex_fetch_desc d = {
         .width = AGX3_TEX_W32,
         .mask = 15,
         .slot = 1,
      };

      for (unsigned v = 0; v < 128; v += 2) {
         d.dest = v;
         CHECK(roundtrip(enc, &d) == 0);
      }
      d.width = AGX3_TEX_W16;
      for (unsigned v = 0; v < 128; ++v) {
         d.dest = v;
         CHECK(roundtrip(enc, &d) == 0);
      }
      d.dest = 0;
      d.width = AGX3_TEX_W32;
      for (unsigned v = 0; v <= 60; v += 4) {
         d.coord = v;
         CHECK(roundtrip(enc, &d) == 0);
      }
      d.coord = 0;
      for (unsigned v = 0; v < 16; ++v) {
         d.texture = v;
         CHECK(roundtrip(enc, &d) == 0);
      }
      d.texture = 0;
      for (unsigned v = 1; v <= (ei ? 8 : 6); ++v) {
         d.slot = v;
         CHECK(roundtrip(enc, &d) == 0);
      }
      d.slot = 1;
      for (unsigned v = 0; v < (ei ? 7 : 6); ++v) {
         d.wait_mask = 1u << v;
         CHECK(roundtrip(enc, &d) == 0);
      }
      d.wait_mask = 0;
      d.mask = 1;
      CHECK(roundtrip(enc, &d) == 0);
      d.mask = 15;
      d.coord_kill = true;
      CHECK(roundtrip(enc, &d) == 0);
      d.final = true;
      CHECK(roundtrip(enc, &d) == 0);
   }
   return 0;
}

static bool
rejected_and_cleared(enum agx_isa_encoding enc,
                     const struct agx3_tex_fetch_desc *d)
{
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length = 0xff;
   memset(bytes, 0xff, sizeof(bytes));
   if (agx3_tex_encode_fetch(enc, d, bytes, &length) || length != 0)
      return false;
   for (unsigned i = 0; i < sizeof(bytes); ++i)
      if (bytes[i] != 0)
         return false;
   return true;
}

static int
test_fail_closed(void)
{
   struct agx3_tex_fetch_desc d = {
      .width = AGX3_TEX_W32, .mask = 15, .slot = 1,
   };
   struct agx3_tex_fetch_desc out;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length;

   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX2, &d));
   d.width = (enum agx3_tex_width)0;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3B, &d));
   d.width = AGX3_TEX_W32;
   d.dest = 1;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3B, &d));
   d.dest = 128;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3C, &d));
   d.dest = 0;
   d.coord = 2;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3B, &d));
   d.coord = 64;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3C, &d));
   d.coord = 0;
   d.mask = 0;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3B, &d));
   d.mask = 15;
   d.texture = 16;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3C, &d));
   d.texture = 0;
   d.slot = 7;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3B, &d));
   d.slot = 9;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3C, &d));
   d.slot = 1;
   d.wait_mask = 3;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3B, &d));
   d.wait_mask = 0x80;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3C, &d));

   d.wait_mask = 0;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, bytes, length - 1,
                                &out));
   CHECK(!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, bytes, length + 1,
                                &out));
   bytes[12] ^= 0x80; /* reserved/fixed selector bit */
   CHECK(!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, bytes, length, &out));
   CHECK(!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, NULL, length, &out));
   CHECK(!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, bytes, length, NULL));
   CHECK(!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX2, bytes, 0, &out));
   return 0;
}

static int
hex_eq(const uint8_t *bytes, uint8_t len, const char *hex)
{
   char got[2 * (AGX3_TEX_MAX_LENGTH > AGX3_TEX_COPY_MAX_LENGTH
                    ? AGX3_TEX_MAX_LENGTH
                    : AGX3_TEX_COPY_MAX_LENGTH) +
            1] = {0};
   for (unsigned i = 0; i < len; ++i)
      snprintf(got + 2 * i, 3, "%02x", bytes[i]);
   if (strcmp(got, hex) != 0) {
      fprintf(stderr, "got %s expected %s\n", got, hex);
      return 0;
   }
   return 1;
}

/*
 * Byte-exact replay of Apple's filtered-sample encodings. Kernel witnesses
 * (explicit level) are from research/gpu/isa-re/gen-x/texops/corpus; the
 * implicit-LOD fragment witnesses (ts_auto / fs_bufcoord) were compiled from
 * fragment render pipelines with Apple's offline translator
 * (MetalToolchain v27.1.5194.15) and decode with the same tables. AGX3b
 * bytes are g16g (byte-identical on g15g); AGX3c bytes are g17s. The G17P
 * `t_base` bytes are also the exact sample in an A18 Pro vertex pipeline with
 * a VBO plus texture0/sampler0, proving that the explicit-level instruction
 * itself is shared across vertex and compute/fragment stages.
 */
static int
test_sample_witnesses(void)
{
   struct svec {
      const char *name;
      struct agx3_tex_sample_desc d;
      const char *hex_b, *hex_c;
   };
   static const struct svec v[] = {
      {"t_base",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 0, .sampler = 0, .lod = AGX3_TEX_LOD_LEVEL, .lod_x64 = 0,
        .slot = 1, .wait_mask = 1, .coord_kill = true, .final = true},
       "05800cb8b0000900000010000100",
       "0f041200418060290100010000000000008000000000"},
      {"x_mask_x",
       {.dest = 2, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "0d800ca090000900000010000100",
       "1f041200418000210100010000000000008000000000"},
      {"x_mask_xy",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 3,
        .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800ca8b0000900000010000100", NULL},
      {"x_mask_zw",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 12,
        .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb890000900000050000100",
       "0f041200418060210300010000000000008000000000"},
      {"x_mask_w",
       {.dest = 2, .coord = 0, .width = AGX3_TEX_W32, .mask = 8,
        .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "0d800cb890000900000010000100", NULL},
      {"l_lod2 level(2.0)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .lod_x64 = 128, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b0000900000010080100",
       "0f041200418060290100010080000000008000000000"},
      {"k_lod7q level(7.25)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .lod_x64 = 464, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b00009800000101c0100",
       "0f0412004180602901000100d0010000008000000000"},
      {"L_smp2x sampler 1",
       {.dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .sampler = 1, .lod = AGX3_TEX_LOD_LEVEL, .slot = 2,
        .coord_kill = true, .final = true},
       "25800cb830240900000110000100",
       "4f041200c98160090100010000000000008000000000"},
      {"L_tex2x texture 1",
       {.dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 1, .lod = AGX3_TEX_LOD_LEVEL, .slot = 2,
        .coord_kill = true, .final = true},
       "25800cb830240900800010000100", NULL},
      {"Ls_slots4 #1",
       {.dest = 16, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .slot = 2, .wait_mask = 1u << 1,
        .coord_kill = true},
       "45800c3830250900000010000100",
       "8f041200c90060490100010000000000008000000000"},
      {"Ls_slots4 #2",
       {.dest = 24, .coord = 4, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 1, .lod = AGX3_TEX_LOD_LEVEL, .slot = 2,
        .wait_mask = 1u << 2, .coord_kill = true},
       "65820c38b0250900800010000100",
       "cf041204c90068690100010000000000008000000000"},
      {"Ls_slots4 #3",
       {.dest = 8, .coord = 8, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 2, .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
        .coord_kill = true},
       "258c0c38b0000900000010000100",
       "4f041208411060290100010000000000008000000000"},
      {"Ls_slots4 #4",
       {.dest = 0, .coord = 12, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 3, .lod = AGX3_TEX_LOD_LEVEL, .slot = 3,
        .wait_mask = 1u << 2, .coord_kill = true, .final = true},
       "058e0cb8b0490900800010000100",
       "0f04120c5190686b0100010000000000008000000000"},
      {"ts_auto implicit LOD",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b0000000010010000100",
       "0f061600418060290118010000000000008000000000"},
      /* Apple's G17P texture.sample offset differential sets f141/f51 and
       * packs signed x/y nibbles in f43. These three result shapes prove the
       * offset fields are orthogonal to both width and mask:
       *
       *   f32x4 plain       05800cb8b0000000010010001200
       *   f32x4 +(0, 0)     05800cb8b0000000010010001200
       *   f32x4 +(1,-1)     05800cb8b000010201881000120e
       *   f16x4 +(1,-1)     35000cb8b000010201881000120e
       *   f32x1 +(1,-1)     1d800ca09000010201881000120e
       *
       * The expected bytes below retain this encoder's hardware-qualified
       * descriptor profile (f83=0/f112=4); only that independent profile
       * differs from Apple's fragment bytes above.
       */
      {"ts_auto offset +(1,-1)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .offset = true,
        .offset_xy = 0xf1},
       "05800cb8b000010201881000010e", NULL},
      {"ts_auto offset half +(1,-1)",
       {.dest = 6, .coord = 0, .width = AGX3_TEX_W16, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .offset = true,
        .offset_xy = 0xf1},
       "35000cb8b000010201881000010e", NULL},
      {"ts_auto offset x-only result",
       {.dest = 6, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .offset = true,
        .offset_xy = 0xf1},
       "1d800ca09000010201881000010e", NULL},
      {"ts_array direct register layer implicit LOD",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .array = true,
        .array_layer = 22},
       "05800cb8b0000db4010010001200", NULL},
      {"texture3d level(2.0)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .lod_x64 = 128, .slot = 1,
        .wait_mask = 1, .coord_kill = true, .final = true,
        .three_d = true},
       "05800cb8b0003900000010080100", NULL},
      {"native cube implicit LOD",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .cube = true},
       "05800cb8b0001000010010001200", NULL},
      {"native cube level(2.0)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .lod_x64 = 128, .slot = 1,
        .wait_mask = 1, .coord_kill = true, .final = true, .cube = true},
       "05800cb8b0001300010010081200", NULL},
      {"native cube register level",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_REGISTER, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .cube = true},
       "05800cb8b0000003010010001200", NULL},
      {"native cube-array dynamic layer implicit LOD",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .array = true, .cube = true,
        .array_layer = 22},
       "05800cb8b00013b5010010001200", NULL},
      {"native cube-array dynamic layer level zero",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .array = true, .cube = true,
        .array_layer = 22},
       "05800cb8b00007b7010010001200", NULL},
      {"native cube comparison",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .cube = true, .shadow = true},
       "05800ca090003000010000000100", NULL},
      {"native cube-array comparison",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .array = true, .cube = true,
        .array_layer = 6, .shadow = true},
       "05800ca090003004010000000100", NULL},
      /*
       * Implicit LOD plus a constant bias.  Apple's own fragment samples of
       * t.sample(s, uv, bias(k)) differ from its plain t.sample(s, uv) in
       * exactly two places -- byte 6 gains 0x07 (f37 = 1, f139 = 1) and the
       * 12-bit x64 field at bytes 7/11 takes k * 64 -- so these rows are the
       * implicit-LOD witness above carrying that same delta:
       *
       *   t.sample(s, uv)              05 80 0c b8 b0 00 00 00 01 00 10 00 12 00
       *   t.sample(s, uv, bias(0.25))  05 80 0c b8 b0 00 07 80 01 00 10 00 12 00
       *   t.sample(s, uv, bias(0.5))   05 80 0c b8 b0 00 07 00 01 00 10 02 12 00
       *   t.sample(s, uv, bias(1.0))   05 80 0c b8 b0 00 07 00 01 00 10 04 12 00
       */
      {"bias(0.25)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS, .lod_x64 = 16, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b0000780010010000100", NULL},
      {"bias(0.5)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS, .lod_x64 = 32, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b0000700010010020100", NULL},
      {"bias(1.0)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS, .lod_x64 = 64, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b0000700010010040100", NULL},
      {"bias(31.0) saturating field",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS, .lod_x64 = 1984, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b00007000100107c0100", NULL},
   };

   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length;
   for (unsigned i = 0; i < sizeof(v) / sizeof(v[0]); ++i) {
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &v[i].d, bytes,
                                   &length));
      CHECK(length == 14 && hex_eq(bytes, length, v[i].hex_b));
      if (v[i].hex_c != NULL) {
         CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &v[i].d, bytes,
                                      &length));
         CHECK(length == 22 && hex_eq(bytes, length, v[i].hex_c));
      }
   }

   /*
    * Apple's f83=4/f112=8 sample is a full fragment ABI, not a stage bit.
    * The A18 Pro hardware matrix with this otherwise identical implicit-LOD
    * oracle was decisive:
    *
    *    f83/f112   result
    *       0/4     exact mip 1 green (driver protocol)
    *       4/4     fixed half-gray
    *       0/8     mip 0 red
    *       4/8     fixed half-gray
    *
    * Keep both complete encodings byte-exact here.  Production descriptors
    * select DRIVER by leaving sample_abi zero; only raw Apple witness tests
    * may opt into APPLE_FRAGMENT until its paired descriptor, coordinate,
    * staging, and derivative ABI is modeled end to end.
    */
   struct agx3_tex_sample_desc abi_sample = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
      .coord_kill = true, .final = true,
   };
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &abi_sample,
                                bytes, &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0000000010010000100"));

   abi_sample.sample_abi = AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &abi_sample,
                                bytes, &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0000000010010001200"));
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &abi_sample,
                                 bytes, &length));

   /* Fresh Metal 32023.917 argument-buffer witness: uniform halves20/24
    * carry the texture/sampler pair under the same complete Apple ABI. */
   abi_sample.texture_kind = AGX3_TEX_OPERAND_UNIFORM;
   abi_sample.texture_value = 20;
   abi_sample.sampler_kind = AGX3_TEX_OPERAND_UNIFORM;
   abi_sample.sampler_value = 24;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &abi_sample,
                                bytes, &length));
   CHECK(length == 20 &&
         hex_eq(bytes, length,
                "05a00cb8b8000000930390011200802000000000"));

   abi_sample.sample_abi = (enum agx3_tex_sample_abi)2;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &abi_sample,
                                 bytes, &length));

   /* The bias form is G17P-only, refuses a value past the 12-bit field, and
    * refuses a zero bias, which Apple emits as the plain implicit sample. */
   struct agx3_tex_sample_desc bad = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_BIAS, .lod_x64 = 32, .slot = 1, .wait_mask = 1,
      .coord_kill = true, .final = true,
   };
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &bad, bytes,
                                 &length));
   bad.lod_x64 = 0;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   bad.lod_x64 = 0x1000;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   bad.lod_x64 = 32;
   bad.array = true;
   bad.array_layer = 22;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b00005b7010010001202"));

   bad = (struct agx3_tex_sample_desc){
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
      .coord_kill = true, .final = true, .three_d = true,
   };
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &bad, bytes,
                                 &length));
   bad.array = true;
   bad.array_layer = 2;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   bad.array = false;
   bad.array_layer = 0;
   bad.lod = AGX3_TEX_LOD_GRADIENT;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   bad.lod = AGX3_TEX_LOD_AUTO;
   bad.gather = AGX3_TEX_GATHER_R;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   return 0;
}

/*
 * Explicit gradients and the per-lane LOD bias.
 *
 * The Apple bytes quoted here were regenerated with
 * `xcrun metal-nt -arch applegpu_g17p` (Metal toolchain 27.1) and decomposed
 * with the production compose tables, which only report a field vector that
 * re-composes to the same bytes. Sources and native objects are in
 * .tmp/g17p-texgrad-evidence (fragment shapes) and the compute kernel
 * `t.sample(s, g[0], gradient2d(g[1], g[2]))`.
 *
 * The gradient sample selects f139 = 5 and is the one sample shape that does
 * *not* set the fragment marker f156: the four staged scalars are the
 * derivative the marker would otherwise request. Apple's compute kernel
 * stages six scalars with the compact copy form, all naming one staging slot,
 * into banks 0, 2, 8, 10, 12 and 14 from six consecutive source registers:
 *
 *    copy 0b10210c   bank  0 <- r16      copy 5b16210c   bank 10 <- r22
 *    copy 1b12210c   bank  2 <- r18      copy 6b18210c   bank 12 <- r24
 *    copy 4b14210c   bank  8 <- r20      copy 7b1a210c   bank 14 <- r26
 *    smp  05800cb8b0000401000010000100
 *
 * which is the bank order, the shared slot and the consecutive-source layout
 * this encoder emits. Which gradient component lands in which bank is pinned
 * independently by the fragment kernels that pass literal gradients, where
 * the staged value is the constant itself: gradient2d((0.25, 0.5),
 * (0.75, 1.0)) loads 0.25 into bank 8, 0.75 into bank 10, 0.5 into bank 12
 * and 1.0 into bank 14, so the interleave is (ddx.x, ddy.x, ddx.y, ddy.y).
 * The (1, 2)/(4, 8) and (16, 32)/(64, 128) pairs agree.
 *
 * The per-lane bias is the per-lane level with one field moved. Apple's
 * fragment pair is byte-identical apart from bytes 6 and 7:
 *
 *    t.sample(s, uv, level(in.uv.x))  05800cb8b0000001010010001200   f139 = 4
 *    t.sample(s, uv, bias(in.uv.x))   05800cb8b0000400010010001200   f139 = 3
 *
 * Both set f156. Those two carry the fragment-only f83 = 4 / f112 = 8
 * descriptor-operand profile that this encoder does not emit, so only the
 * delta between them is asserted here, on the encoder's own operand profile.
 */
static int
test_gradient_and_register_bias(void)
{
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length;

   /* Apple's own compute gradient sample, reproduced byte for byte:
    * immediate texture and sampler, coordinates in bank 0, waiting on the
    * staging slot the six copies published. */
   const struct agx3_tex_sample_desc grad_apple = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_GRADIENT, .slot = 1, .wait_mask = 1u << 0,
      .coord_kill = true, .final = true,
   };
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &grad_apple, bytes,
                                &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0000401000010000100"));

   /* The same sample waiting on a different staging slot. */
   const struct agx3_tex_sample_desc grad = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_GRADIENT, .slot = 1, .wait_mask = 1u << 2,
      .coord_kill = true, .final = true,
   };
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &grad, bytes,
                                &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0010401000010000100"));

   /* The same gradient sample reading its texture and sampler out of the
    * bindless heap. Apple's own argument-buffer compute kernel
    * `ts.t.sample(ts.s, g[0], gradient2d(g[1], g[2]))` binds them at uniform
    * halves 20 and 24 and emits the identical six staging copies, so the
    * gradient mode is orthogonal to the descriptor operand class and there is
    * no separate heap gradient form to ground. Byte-exact. */
   struct agx3_tex_sample_desc grad_heap = grad_apple;
   grad_heap.texture_kind = AGX3_TEX_OPERAND_UNIFORM;
   grad_heap.texture_value = 20;
   grad_heap.sampler_kind = AGX3_TEX_OPERAND_UNIFORM;
   grad_heap.sampler_value = 24;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &grad_heap, bytes,
                                &length));
   CHECK(length == 20 &&
         hex_eq(bytes, length,
                "05a00cb8b8000401920390010100802000000000"));

   /* The six staging copies this encoder emits, in Apple's bank order. The
    * source-operand class is the packer's own witnessed staging form (f16,
    * f23 = 2), not the compact f17 / f23 = 3 form that particular Apple
    * kernel happened to use, so these are round-trips of the shared helper
    * every already-qualified sample path uses -- what is asserted here is the
    * bank assignment and the shared slot. */
   struct {
      uint16_t dest, src, wait;
      const char *hex;
   } stage[] = {
      {0, 8, 1u << 5, "0b092f00420000880000"},
      {2, 10, 0, "1b0b2f00420000080000"},
      {AGX3_TEX_GRADIENT_BANK_DDX_X, 4, 1u << 0, "4b052f00422000080000"},
      {AGX3_TEX_GRADIENT_BANK_DDY_X, 0, 1u << 1, "5b012f00424000080000"},
      {AGX3_TEX_GRADIENT_BANK_DDX_Y, 6, 0, "6b072f00420000080000"},
      {AGX3_TEX_GRADIENT_BANK_DDY_Y, 2, 0, "7b032f00420000080000"},
   };
   for (unsigned i = 0; i < sizeof(stage) / sizeof(stage[0]); ++i) {
      uint8_t copy[AGX3_TEX_COPY_MAX_LENGTH];
      CHECK(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, stage[i].dest,
                                      stage[i].src, 3, stage[i].wait, false,
                                      true, copy, &length));
      CHECK(length == 10 && hex_eq(copy, length, stage[i].hex));
   }

   /* The per-lane level, byte-exact, and the per-lane bias derived from it. */
   struct agx3_tex_sample_desc level_reg = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_REGISTER, .slot = 1, .wait_mask = 1u << 1,
      .coord_kill = true, .final = true,
   };
   uint8_t level_bytes[AGX3_TEX_MAX_LENGTH], bias_bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t level_len, bias_len;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &level_reg,
                                level_bytes, &level_len));
   CHECK(level_len == 14 &&
         hex_eq(level_bytes, level_len, "05800cb830010001010010000100"));

   struct agx3_tex_sample_desc bias_reg = level_reg;
   bias_reg.lod = AGX3_TEX_LOD_BIAS_REGISTER;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bias_reg, bias_bytes,
                                &bias_len));
   CHECK(bias_len == level_len);

   /* Exactly Apple's own delta: bytes 6 and 7 move (0x00, 0x01) ->
    * (0x04, 0x00), nothing else changes. */
   for (unsigned i = 0; i < level_len; ++i) {
      if (i == 6) {
         CHECK(level_bytes[i] == 0x00 && bias_bytes[i] == 0x04);
      } else if (i == 7) {
         CHECK(level_bytes[i] == 0x01 && bias_bytes[i] == 0x00);
      } else {
         CHECK(level_bytes[i] == bias_bytes[i]);
      }
   }

   /* The immediate bias field f29 is a *signed* 12-bit LOD in 1/64 steps.
    * Sweeping Apple's own `t.sample(s, uv, bias(k))` over these k gives
    * exactly k * 64 in two's complement; the fragment kernels differ from
    * this encoder only in the f83 = 4 / f112 = 8 descriptor-operand profile,
    * which is the trailing byte. Apple saturates at 2047 (bias 32, 63) and
    * at 2048 (bias -33, -64), so a minifloat immediate, which spans
    * [-31, 31], always lands inside the representable range.
    */
   const struct {
      int x64;
      const char *hex;
   } bias_sweep[] = {
      {   16, "05800cb8b0000780010010000100" },   /* bias  +0.015625 * 16 */
      {   32, "05800cb8b0000700010010020100" },   /* bias  +0.5           */
      {   64, "05800cb8b0000700010010040100" },   /* bias  +1             */
      {   96, "05800cb8b0000700010010060100" },   /* bias  +1.5           */
      {  128, "05800cb8b0000700010010080100" },   /* bias  +2             */
      {  256, "05800cb8b0000700010010100100" },   /* bias  +4             */
      {    1, "05800cb8b0000708010010000100" },   /* bias  +1/64          */
      { 1984, "05800cb8b00007000100107c0100" },   /* bias  +31            */
      { 4095, "05801cb8b00007f80100107e0100" },   /* bias  -1/64          */
      { 4064, "05801cb8b00007000100107e0100" },   /* bias  -0.5           */
      { 4032, "05801cb8b00007000100107c0100" },   /* bias  -1             */
      { 3968, "05801cb8b0000700010010780100" },   /* bias  -2             */
      { 3840, "05801cb8b0000700010010700100" },   /* bias  -4             */
      { 2112, "05801cb8b0000700010010040100" },   /* bias  -31            */
   };
   for (unsigned i = 0; i < sizeof(bias_sweep) / sizeof(bias_sweep[0]); ++i) {
      struct agx3_tex_sample_desc b = {
         .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
         .lod = AGX3_TEX_LOD_BIAS, .slot = 1, .wait_mask = 1u << 0,
         .coord_kill = true, .final = true,
         .lod_x64 = (uint16_t)bias_sweep[i].x64,
      };
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &b, bytes,
                                   &length));
      CHECK(length == 14 && hex_eq(bytes, length, bias_sweep[i].hex));
   }

   /* textureGather: f144 = 1, the component in f145, and the f74 mask-group
    * bit every other sample sets is cleared. Apple's five fragment kernels
    * differ from these only in the f83 = 4 / f112 = 8 descriptor-operand
    * profile, which is the trailing byte:
    *
    *    t.gather(s, uv, int2(0), component::x)  05800ca490000000010000001200
    *    t.gather(s, uv, int2(0), component::y)  05800cac90000000010000001200
    *    t.gather(s, uv, int2(0), component::z)  05800cb490000000010000001200
    *    t.gather(s, uv, int2(0), component::w)  05800cbc90000000010000001200
    *
    * `t.gather(s, uv)` with no component is byte-identical to component::x.
    */
   const char *const gathers[] = {
      "05800ca490000000010000000100", "05800cac90000000010000000100",
      "05800cb490000000010000000100", "05800cbc90000000010000000100",
   };
   for (unsigned g = AGX3_TEX_GATHER_R; g <= AGX3_TEX_GATHER_A; ++g) {
      struct agx3_tex_sample_desc gd = {
         .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
         .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1u << 0,
         .coord_kill = true, .final = true,
         .gather = (enum agx3_tex_gather)g,
      };
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &gd, bytes,
                                   &length));
      CHECK(length == 14 &&
            hex_eq(bytes, length, gathers[g - AGX3_TEX_GATHER_R]));

      /* Only the plain 2D implicit gather is witnessed. */
      struct agx3_tex_sample_desc bad_gather = gd;
      bad_gather.array = true;
      CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad_gather,
                                    bytes, &length));
      bad_gather = gd;
      bad_gather.lod = AGX3_TEX_LOD_BIAS;
      bad_gather.lod_x64 = 32;
      CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad_gather,
                                    bytes, &length));
      CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &gd, bytes,
                                    &length));
   }

   /* Apple's depth2d sample_compare differs from the ordinary sample by
    * f143=1, f75=1 and the absence of f74.  Normalize Apple's destination
    * half6 to half0; the trailing profile byte is Mesa's hardware-qualified
    * fragment profile. */
   struct agx3_tex_sample_desc shadow = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1u << 0,
      .coord_kill = true, .final = true, .shadow = true,
   };
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &shadow, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800ca090002000010000000100"));
   struct agx3_tex_sample_desc bad_shadow = shadow;
   bad_shadow.mask = 3;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad_shadow, bytes,
                                 &length));
   /* applegpu-nt 32023.917 / apple-g17p. The array comparison clears the
    * ordinary array register/profile fields and selects packed layer f77=2. */
   struct agx3_tex_sample_desc shadow_array = shadow;
   shadow_array.array = true;
   shadow_array.array_layer = 6;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &shadow_array, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800ca090002004010000000100"));
   /* applegpu-nt 32023.917: cube comparison adds the f76=4 dimension;
    * cube-array comparison also selects the packed layer with f77=2. Both
    * deliberately retain the driver's comparison operand profile. */
   struct agx3_tex_sample_desc shadow_cube = shadow;
   shadow_cube.cube = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &shadow_cube, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800ca090003000010000000100"));
   struct agx3_tex_sample_desc shadow_cube_array = shadow_array;
   shadow_cube_array.cube = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,
                                &shadow_cube_array, bytes, &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800ca090003004010000000100"));
   bad_shadow = shadow_array;
   bad_shadow.array_layer = 3;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad_shadow, bytes,
                                 &length));
   bad_shadow = shadow_array;
   bad_shadow.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   bad_shadow.texture_value = 2;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad_shadow, bytes,
                                 &length));
   bad_shadow = shadow_array;
   bad_shadow.sample_abi = AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad_shadow, bytes,
                                 &length));
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &shadow, bytes,
                                 &length));

   /* Both forms are AGX3b-only and neither carries an immediate LOD. */
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &grad, bytes,
                                 &length));
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &bias_reg, bytes,
                                 &length));
   struct agx3_tex_sample_desc bad = grad;
   bad.lod_x64 = 32;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   bad = grad;
   bad.array = true;
   bad.array_layer = 22;
   bad.wait_mask = 1;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0000fb5000010001200"));
   bad = bias_reg;
   bad.lod_x64 = 32;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   bad = bias_reg;
   bad.array = true;
   bad.array_layer = 22;
   bad.wait_mask = 1;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0000db5010010001200"));

   struct agx3_tex_sample_desc array_level = grad_apple;
   array_level.array = true;
   array_level.array_layer = 22;
   array_level.lod = AGX3_TEX_LOD_LEVEL;
   array_level.lod_x64 = 128;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &array_level, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0008db6010010001208"));
   array_level.lod = AGX3_TEX_LOD_REGISTER;
   array_level.lod_x64 = 0;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &array_level, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0008fb4010010001200"));

   bad = grad_apple;
   bad.array = true;
   bad.array_layer = 3;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   bad.array_layer = 256;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   return 0;
}

/*
 * The coordinate-staging copy, replayed against the kernel witnesses
 * (t_base, x_coordreg, Ls_slots4) and the fragment fs_bufcoord witnesses.
 * The wait bit names the producer's scoreboard slot (f30 + slot - 1); a
 * slot above 1 or an AGX3c wait past slot 6 escalates the encoding point,
 * exactly as Apple's own choice does.
 */
static int
test_copy_witnesses(void)
{
   struct cvec {
      const char *name;
      enum agx_isa_encoding enc;
      uint16_t dest, src;
      uint8_t slot;
      uint16_t wait_mask;
      const char *hex;
   };
   static const struct cvec v[] = {
      {"b x_coordreg r0<-r0 wait s6", AGX_ISA_ENCODING_AGX3B, 0, 0, 1,
       1u << 5, "0b0029c4"},
      {"b x_coordreg r1<-r1", AGX_ISA_ENCODING_AGX3B, 2, 2, 1, 0,
       "1b022904"},
      {"b t_base r1<-r0", AGX_ISA_ENCODING_AGX3B, 2, 0, 1, 0, "1b002904"},
      {"b Ls #1 r0<-r4 slot2 wait s6", AGX_ISA_ENCODING_AGX3B, 0, 8, 2,
       1u << 5, "0b092f00420000840000"},
      {"b Ls #2 r1<-r5 slot2", AGX_ISA_ENCODING_AGX3B, 2, 10, 2, 0,
       "1b0b2f00420000040000"},
      {"b Ls #3 r2<-r6 slot3", AGX_ISA_ENCODING_AGX3B, 4, 12, 3, 0,
       "2b0d2f00420000080000"},
      {"b Ls #5 r4<-r0 wait s1", AGX_ISA_ENCODING_AGX3B, 8, 0, 1, 1,
       "4b002924"},
      {"b Ls #6 r5<-r1", AGX_ISA_ENCODING_AGX3B, 10, 2, 1, 0, "5b022904"},
      {"b Ls #7 r6<-r2 slot3", AGX_ISA_ENCODING_AGX3B, 12, 4, 3, 0,
       "6b052f00420000080000"},
      {"c fs_bufcoord r0<-r0 wait s8", AGX_ISA_ENCODING_AGX3C, 0, 0, 1,
       1u << 7, "2b0007022680a402"},
      {"c fs_bufcoord r1<-r1", AGX_ISA_ENCODING_AGX3C, 2, 2, 1, 0,
       "1b022904"},
      {"c t_base r1<-r0", AGX_ISA_ENCODING_AGX3C, 2, 0, 1, 0, "1b002904"},
      {"c Ls #1 r0<-r4 slot2 wait s8", AGX_ISA_ENCODING_AGX3C, 0, 8, 2,
       1u << 7, "2b0007122681a402"},
      {"c Ls #2 r1<-r5 slot2", AGX_ISA_ENCODING_AGX3C, 2, 10, 2, 0,
       "330207122681a402"},
      {"c Ls #3 r2<-r6 slot3", AGX_ISA_ENCODING_AGX3C, 4, 12, 3, 0,
       "2300071a2682a40a"},
      {"c Ls #5 r4<-r0 wait s1", AGX_ISA_ENCODING_AGX3C, 8, 0, 1, 1,
       "4b002924"},
      {"c Ls #7 r6<-r2 slot3", AGX_ISA_ENCODING_AGX3C, 12, 4, 3, 0,
       "2300070a2682a41a"},
   };

   uint8_t bytes[AGX3_TEX_COPY_MAX_LENGTH];
   uint8_t length;
   for (unsigned i = 0; i < sizeof(v) / sizeof(v[0]); ++i) {
      CHECK(agx3_tex_stage_copy(v[i].enc, v[i].dest, v[i].src, v[i].slot,
                                v[i].wait_mask, bytes, &length));
      if (!hex_eq(bytes, length, v[i].hex)) {
         fprintf(stderr, "copy witness: %s\n", v[i].name);
         return 1;
      }
      checks++;
   }
   return 0;
}

static int
test_sample_fail_closed(void)
{
   const struct agx3_tex_sample_desc good = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length;
   struct agx3_tex_sample_desc d;

   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &good, bytes,
                                &length));
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX2, &good, bytes,
                                 &length));

   d = good;
   d.mask = 0; /* empty samples have no native result */
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.sampler = 16;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.lod_x64 = 64; /* AUTO carries no level operand */
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.lod = AGX3_TEX_LOD_LEVEL;
   d.lod_x64 = 0x1000;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.coord = 2;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.wait_mask = 3;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.offset = true;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.offset_xy = 0xf1;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   d.lod = AGX3_TEX_LOD_LEVEL;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.lod = AGX3_TEX_LOD_AUTO;
   d.array = true;
   d.array_layer = 2;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.array = false;
   d.array_layer = 0;
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.offset_xy = 0xf1;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));

   /* Stage-copy edges: odd registers, bad slots, wide waits. */
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 1, 0, 1, 0, bytes,
                              &length));
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, 3, 1, 0, bytes,
                              &length));
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, 0, 0, 0, bytes,
                              &length));
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, 0, 7, 0, bytes,
                              &length));
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, 0, 1, 1u << 6,
                              bytes, &length));
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3C, 0, 0, 9, 0, bytes,
                              &length));
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, 0, 1, 3, bytes,
                              &length));
   return 0;
}

/*
 * The 2D-array texel fetch. Both vectors are Apple's own bytes, from
 * `texture2d<float, access::read>::read()` and
 * `texture2d_array<float, access::read>::read()` compiled by
 * `metal-nt -arch applegpu_g16g`; they differ in the array selector alone.
 */
static int
test_array_fetch(void)
{
   static const uint8_t plain_2d[14] = {0x05, 0x80, 0x0c, 0xb8, 0x30, 0x01,
                                        0x17, 0x01, 0x00, 0x00, 0x00, 0x00,
                                        0x01, 0x00};
   static const uint8_t array_2d[14] = {0x05, 0x80, 0x0c, 0xb8, 0xb0, 0x00,
                                        0x97, 0x01, 0x00, 0x00, 0x00, 0x00,
                                        0x01, 0x00};

   const struct agx3_tex_fetch_desc plain = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .texture = 0, .slot = 1, .wait_mask = 1u << 1, .coord_kill = true,
      .final = true, .array = false,
   };
   const struct agx3_tex_fetch_desc arr = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .texture = 0, .slot = 1, .wait_mask = 1u << 0, .coord_kill = true,
      .final = true, .array = true,
   };

   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length = 0;

   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &plain, bytes,
                               &length));
   CHECK(length == 14);
   CHECK(memcmp(bytes, plain_2d, 14) == 0);

   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &arr, bytes, &length));
   CHECK(length == 14);
   CHECK(memcmp(bytes, array_2d, 14) == 0);

   /* The array flag survives a decode, and the two forms stay distinct. */
   struct agx3_tex_fetch_desc back;
   CHECK(agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, array_2d, 14, &back));
   CHECK(back.array);
   CHECK(same(&back, &arr));
   CHECK(agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, plain_2d, 14, &back));
   CHECK(!back.array);
   CHECK(same(&back, &plain));

   return 0;
}


/*
 * Bindless operand kinds. The texture and sampler each select one of three
 * consecutive fields by kind (95/96/97 and 124/125/126), and immediate vs
 * non-immediate also selects the encoding point: ep3 for immediate, ep4 for
 * register and uniform. All three kinds are witnessed byte-exactly from
 * Apple shaders captured on an A18 Pro; see test-agx3-smp-witness.c.
 */
static int
test_bindless_operand_kinds(void)
{
   struct agx3_tex_fetch_desc d;
   uint8_t out[AGX3_TEX_MAX_LENGTH];
   uint8_t len;

#define FETCH_BASE()                                                           \
   do {                                                                        \
      memset(&d, 0, sizeof(d));                                                \
      d.dest = 8;                                                              \
      d.coord = 16;                                                            \
      d.width = 1;                                                             \
      d.mask = 15;                                                             \
      d.slot = 1;                                                              \
      d.array = true;                                                          \
      len = 0;                                                                 \
   } while (0)

   /* Immediate operands keep the 14-byte ep3 form. A caller that never heard
    * of operand kinds leaves them zero and must encode exactly as before. */
   FETCH_BASE();
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   CHECK(len == 14);

   /* Register descriptors: the ep4 form, 20 bytes. */
   FETCH_BASE();
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = 0;
   d.sampler_value = 4;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   CHECK(len == 20);

   /* Uniform descriptors: also ep4. */
   FETCH_BASE();
   d.texture_kind = AGX3_TEX_OPERAND_UNIFORM;
   d.sampler_kind = AGX3_TEX_OPERAND_UNIFORM;
   d.texture_value = 16;
   d.sampler_value = 20;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   CHECK(len == 20);

   /* Fresh native read() uses an indirect texture with an immediate
    * sampler index. Each operand has its own immediate/indirect class. */
   FETCH_BASE();
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, out, &len));

   /* Fresh MetalToolchain v27.1.5194.15 vertex argument-buffer read. The
    * dynamic VS object (SHA256 2838ec963f7e...) uses SMP.EP4 with its
    * zero-extended texture heap index in uniform half16 and the native read
    * sampler at immediate zero. Its direct texture0 sibling is the ordinary
    * 14-byte EP3 form; this 20-byte instruction is their only SMP delta. */
   memset(&d, 0, sizeof(d));
   d.dest = 0;
   d.coord = 0;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.slot = 1;
   d.wait_mask = 1;
   d.coord_kill = true;
   d.final = true;
   d.texture_kind = AGX3_TEX_OPERAND_UNIFORM;
   d.texture_value = 16;
   d.sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   CHECK(len == 20 &&
         hex_eq(out, len,
                "05800cb8b0001701900280000100002000000000"));

   /* Only AGX3b is witnessed for the non-immediate kinds. */
   FETCH_BASE();
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, out, &len));

#undef FETCH_BASE
   return 0;
}

/* Fresh offline G17P Metal argument-buffer samples, September 5 2026.
 * Native UMD 0x33dcc8/0x33db40 stores zero-extended heap indices into these
 * 64-bit argument slots; their contents are not descriptor addresses.
 */
static int
test_indexed_sample_witnesses(void)
{
   const uint8_t vertex_level_bytes[] = {
      0x05,0xa0,0x0c,0xb8,0xb8,0x00,0x09,0x00,0x92,0x03,
      0x90,0x01,0x01,0x00,0x80,0x20,0x00,0x00,0x00,0x00};
   const uint8_t vertex_mixed_bytes[] = {
      0x05,0x80,0x0c,0xb8,0xb0,0x00,0x09,0x00,0x90,0x02,
      0x90,0x00,0x01,0x00,0x00,0x20,0x00,0x00,0x00,0x00};
   const uint8_t uniform_bytes[] = {
      0x05,0xe0,0x0c,0xb8,0xb0,0x00,0x00,0x00,0x83,0x03,
      0x90,0x01,0x01,0x00,0x80,0x20,0x00,0x00,0x00,0x00};
   const uint8_t register_bytes[] = {
      0x05,0x80,0x0c,0xb8,0x34,0x00,0x00,0x00,0x01,0x02,
      0x90,0x01,0x01,0x00,0xa0,0x28,0x00,0x00,0x40,0x00};
   const uint8_t uniform_array_bytes[] = {
      0x05,0xc0,0x0c,0xb8,0xbc,0x00,0x0d,0x14,0x93,0x03,
      0x90,0x03,0x12,0x00,0x80,0x20,0x00,0x00,0x00,0x00};
   struct agx3_tex_sample_desc d = {
      .width=AGX3_TEX_W32, .mask=15, .slot=1, .wait_mask=1,
      .coord_kill=true, .final=true, .lod=AGX3_TEX_LOD_AUTO,
      .texture_kind=AGX3_TEX_OPERAND_UNIFORM,
      .sampler_kind=AGX3_TEX_OPERAND_UNIFORM,
      .texture_value=12, .sampler_value=16,
   };
   uint8_t out[AGX3_TEX_MAX_LENGTH], len;
   /* Fresh v27.1 G17P vertex dynamic argument-buffer sample, object SHA256
    * 49cab707568f.... Texture/sampler heap indices are uniform20/uniform24;
    * the explicit vertex level-zero selector is otherwise the direct sample
    * sibling's exact SMP shape. */
   d.lod = AGX3_TEX_LOD_LEVEL;
   d.texture_value = 20;
   d.sampler_value = 24;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(len == sizeof(vertex_level_bytes));
   CHECK(memcmp(out, vertex_level_bytes, len) == 0);
   /* The independently compiled mixed sibling (SHA256 6fa7259810a5...)
    * keeps texture uniform16 indirect while selecting sampler0 immediate.
    * Operand class and size selectors are independent in SMP.EP4. */
   d.texture_value = 16;
   d.sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE;
   d.sampler_value = 0;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(len == sizeof(vertex_mixed_bytes));
   CHECK(memcmp(out, vertex_mixed_bytes, len) == 0);
   d.lod = AGX3_TEX_LOD_AUTO;
   d.sampler_kind = AGX3_TEX_OPERAND_UNIFORM;
   d.texture_value = 12;
   d.sampler_value = 16;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(len==sizeof(uniform_bytes));
   CHECK(memcmp(out,uniform_bytes,len)==0);
   d.cube=true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(hex_eq(out,len,
                "05e00cb8b0001000830390010100802000000000"));
   d.cube=false;
   d.array=true;
   d.array_layer=34;
   d.texture_value=24;
   d.sampler_value=28;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(len==sizeof(uniform_array_bytes));
   CHECK(memcmp(out,uniform_array_bytes,len)==0);
   d.array=false;
   d.array_layer=0;
   d.texture_kind=d.sampler_kind=AGX3_TEX_OPERAND_REGISTER;
   d.texture_value=0; d.sampler_value=4; d.wait_mask=32;
   d.texture_kill=d.sampler_kill=true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(len==sizeof(register_bytes));
   CHECK(memcmp(out,register_bytes,len)==0);
   d.cube=true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(hex_eq(out,len,
                "05800cb834001000010290010100a02800004000"));
   d.cube=false;
   d.array=true;
   d.array_layer=24;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(hex_eq(out,len,
                "05800cb834000dc4010290011200a02800004000"));
   d.array=false;
   d.array_layer=0;
   d.texture_value=1;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   d.texture_value=0; d.texture_kind=AGX3_TEX_OPERAND_UNIFORM;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   return 0;
}

static int
test_sample_mask_domain(void)
{
   /* Native mask_1..mask_15, every nonempty packed result subset. */
   static const uint8_t native[15][14] = {
      {0x05,0x80,0x0c,0xa0,0x90,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa8,0x90,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa8,0xb0,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb0,0x90,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa0,0x90,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa8,0x90,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb0,0xb0,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb8,0x90,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa0,0xb0,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb0,0x90,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa8,0xb0,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb8,0x90,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa0,0xb0,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb0,0xb0,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb8,0xb0,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00}
   };
   for (unsigned mask=1; mask<16; ++mask) {
      struct agx3_tex_sample_desc d={
         .width=AGX3_TEX_W32,.mask=mask,.slot=1,.wait_mask=1,
         .coord_kill=true,.final=true,.lod=AGX3_TEX_LOD_AUTO,
      };
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==14);
      CHECK(memcmp(out,native[mask-1],len)==0);
   }
   return 0;
}

static int
test_indexed_fetch_native(void)
{
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb8,0xb0,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0xe0,0x0c,0xb8,0xb0,0x00,0x17,0x01,0x80,0x03,0x80,0x00,0x01,0x00,0x00,0x20,0x00,0x00,0x00,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=3,.texture_value=12,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb8,0x30,0x00,0x17,0x01,0x00,0x03,0x80,0x00,0x01,0x00,0x00,0x28,0x00,0x00,0x40,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=2,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=32,.coord_kill=true,.array=0,
         .final=1,.texture_kill=1};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0xe0,0x0c,0xb8,0xb0,0x00,0x97,0x01,0x80,0x03,0x80,0x00,0x01,0x00,0x00,0x20,0x00,0x00,0x00,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=3,.texture_value=12,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=1,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0xc0,0x0c,0xb8,0xb0,0x00,0x97,0x01,0x00,0x03,0x80,0x00,0x01,0x00,0x00,0x28,0x00,0x00,0x00,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=2,.texture_value=8,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=1,
         .final=1,.texture_kill=1};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0xe0,0x0c,0x38,0xb1,0x00,0x17,0x01,0x80,0x02,0x80,0x00,0x01,0x00,0x00,0x20,0x00,0x00,0x00,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=3,.texture_value=12,.sampler_value=2,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=0,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa0,0x90,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=1,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa8,0x90,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=2,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa8,0xb0,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=3,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x0d,0x80,0x0c,0xb0,0x90,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=2,.coord=0,.width=AGX3_TEX_W32,
         .mask=4,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa0,0x90,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=5,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa8,0x90,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=6,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb0,0xb0,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=7,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x0d,0x80,0x0c,0xb8,0x90,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=2,.coord=0,.width=AGX3_TEX_W32,
         .mask=8,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa0,0xb0,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=9,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb0,0x90,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=10,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa8,0xb0,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=11,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb8,0x90,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=12,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa0,0xb0,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=13,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb0,0xb0,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=14,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb8,0xb0,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   return 0;
}

static int
test_staging_source_lifetime(void)
{
   /* Fresh complete native read kernels use coordinates both for TEX and
    * later output addressing/XOR. Their staging producers must not discard. */
   const uint8_t golden[2][2][4]={
      {{0x0b,0x08,0x21,0x24},{0x1b,0x0a,0x21,0x44}},
      {{0x0b,0x88,0x21,0x24},{0x1b,0x8a,0x21,0x44}},
   };
   for(unsigned cache=0;cache<2;cache++)for(unsigned xy=0;xy<2;xy++) {
      uint8_t bytes[AGX3_TEX_COPY_MAX_LENGTH],len;
      CHECK(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B,2*xy,8+2*xy,
         1,1u<<xy,cache,false,bytes,&len));
      CHECK(len==4 && !memcmp(bytes,golden[cache][xy],4));
   }
   uint8_t rejected[AGX3_TEX_COPY_MAX_LENGTH], length=0xff;
   memset(rejected,0xa5,sizeof(rejected));
   CHECK(!agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B,0,8,1,0,
      true,true,rejected,&length));
   CHECK(length==0);
   for(unsigned i=0;i<sizeof(rejected);i++)CHECK(rejected[i]==0);
   return 0;
}

static int
test_register_lod(void)
{
   /* Fresh native G17P fragment bodies, float-to-half and retained half LOD. */
   const uint8_t dead[] = {0x17,0x05,0x54,0x06,0,0,0,0x04,0xf0,0xc0,0x08,0};
   const uint8_t live[] = {0x17,0x05,0x56,0x06,0,0x10,0,0x04,0x70,0xc0,0x08,0};
   const uint8_t sample[] = {0x05,0x80,0x0c,0xb8,0xb0,0,0,1,1,0,0x10,0,1,0};
   uint8_t b[AGX3_TEX_MAX_LENGTH], len;
   CHECK(agx3_tex_stage_lod(AGX_ISA_ENCODING_AGX3B,0,1,0,false,true,b,&len));
   CHECK(len == sizeof(dead) && !memcmp(b,dead,len));
   CHECK(agx3_tex_stage_lod(AGX_ISA_ENCODING_AGX3B,8,1,1u<<5,false,false,b,&len));
   CHECK(len == sizeof(live) && !memcmp(b,live,len));
   struct agx3_tex_sample_desc d = {.dest=0,.coord=0,.width=AGX3_TEX_W32,
      .mask=15,.lod=AGX3_TEX_LOD_REGISTER,.slot=1,.wait_mask=1,
      .coord_kill=true,.final=true};
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,b,&len));
   CHECK(len == sizeof(sample) && !memcmp(b,sample,len));
   /* Independently decode the full eight-bit field, including the upper
    * bank. Adjacent CACHE/DISCARD and scoreboard bits must stay intact. */
   for (unsigned src = 0; src < 256; ++src) {
      CHECK(agx3_tex_stage_lod(AGX_ISA_ENCODING_AGX3B,src,1,0,false,true,b,&len));
      CHECK(len == 12);
      CHECK(((b[5] >> 1) | ((b[6] & 1) << 7)) == src);
      b[5] &= 1;
      b[6] &= ~1;
      CHECK(!memcmp(b,dead,sizeof(dead)));
   }
   return 0;
}

/*
 * The texel fetch's level.
 *
 * Three `metal-nt -arch applegpu_g17p` compute kernels differing only in the
 * level argument to texture2d<float>::read(), from one source file
 * (.tmp/g17p-gallium2-evidence/texlod.metal):
 *
 *   read(gid, 0)        f37 = 1, f139 = 2, f29 = 0
 *   read(gid, 2)        f37 = 1, f139 = 2, f29 = 128     -- the level x 64
 *   read(gid, lod[i])   f37 absent, f139 = 4, preceded by a PACK_SMPIR
 *
 * The same delta appears between two fragment programs in texlodfs.metal, so
 * it is not a compute-only shape; those bytes additionally carry the
 * fragment-only f83/f112/f124 selectors this encoder does not model, which is
 * why the vectors here are the compute pair.
 */
static int
test_fetch_lod(void)
{
   const uint8_t lod0[] = {0x05, 0x80, 0x0c, 0xb8, 0xb0, 0x00, 0x17,
                           0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00};
   const uint8_t lod2[] = {0x05, 0x80, 0x0c, 0xb8, 0xb0, 0x00, 0x17,
                           0x01, 0x00, 0x00, 0x00, 0x08, 0x01, 0x00};
   const uint8_t dynamic[] = {0x05, 0x80, 0x0c, 0xb8, 0x30, 0x01, 0x80,
                              0x24, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00};

   struct agx3_tex_fetch_desc d = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .texture = 0, .slot = 1, .wait_mask = 1u << 0,
      .coord_kill = true, .final = true,
   };
   uint8_t b[AGX3_TEX_MAX_LENGTH], len = 0;

   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, b, &len));
   CHECK(len == sizeof(lod0) && !memcmp(b, lod0, len));

   d.lod_x64 = 2 * 64;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, b, &len));
   CHECK(len == sizeof(lod2) && !memcmp(b, lod2, len));

   /* The per-lane form waits on the staging slot the PACK_SMPIR published,
    * which is why Apple's own program moves the wait from f62 to f63. */
   d.lod_x64 = 0;
   d.register_lod = true;
   d.wait_mask = 1u << 1;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, b, &len));
   CHECK(len == sizeof(dynamic) && !memcmp(b, dynamic, len));

   /* An immediate level and a per-lane level are mutually exclusive, and the
    * per-lane form has no AGX3c witness. */
   d.lod_x64 = 64;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, b, &len));
   d.lod_x64 = 0;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, b, &len));
   d.register_lod = false;
   d.lod_x64 = 2048;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, b, &len));

   /* The level staged for a texel fetch is an INTEGER: PACK_SMPIR's f74 is
    * the only field that moves between the fragment fetch and the fragment
    * filtered sample in texlodtype.metal. */
   const uint8_t integer_level[] = {0x17, 0x05, 0x54, 0x06, 0x00, 0x08,
                                    0x00, 0x04, 0xf0, 0xc4, 0x0c, 0x00};
   const uint8_t float_level[] = {0x17, 0x05, 0x54, 0x06, 0x00, 0x0c,
                                  0x00, 0x04, 0xf0, 0xc0, 0x08, 0x00};
   CHECK(agx3_tex_stage_typed_lod(AGX_ISA_ENCODING_AGX3B, 4, 2, 0, false,
                                  true, AGX3_TEX_LEVEL_INTEGER, b, &len));
   CHECK(len == sizeof(integer_level) && !memcmp(b, integer_level, len));
   CHECK(agx3_tex_stage_typed_lod(AGX_ISA_ENCODING_AGX3B, 6, 1, 0, false,
                                  true, AGX3_TEX_LEVEL_FLOAT, b, &len));
   CHECK(len == sizeof(float_level) && !memcmp(b, float_level, len));

   /* The default entry point stays the float form every existing caller
    * relies on. */
   uint8_t plain[AGX3_TEX_MAX_LENGTH], plain_len = 0;
   CHECK(agx3_tex_stage_lod(AGX_ISA_ENCODING_AGX3B, 6, 1, 0, false, true,
                            plain, &plain_len));
   CHECK(plain_len == len && !memcmp(plain, b, len));

   /* No other level type is witnessed. */
   CHECK(!agx3_tex_stage_typed_lod(AGX_ISA_ENCODING_AGX3B, 4, 2, 0, false,
                                   true, (enum agx3_tex_level_kind)1, b,
                                   &len));
   CHECK(!agx3_tex_stage_typed_lod(AGX_ISA_ENCODING_AGX3B, 4, 2, 0, false,
                                   true, (enum agx3_tex_level_kind)4, b,
                                   &len));
   return 0;
}

static int
test_register_array_layer(void)
{
   /* Native direct/uniform sample: discarded r2 publishes on slot 1. */
   const uint8_t native_slot1[] = {
      0x17, 0x05, 0x54, 0x06, 0x00, 0x08,
      0x00, 0x00, 0xf0, 0xc0, 0x04, 0x00,
   };
   /* Native register-indexed sample: discarded r6 publishes on slot 6. */
   const uint8_t native_register[] = {
      0x17, 0x05, 0x54, 0x06, 0x00, 0x18,
      0x00, 0x00, 0xf0, 0xe4, 0x04, 0x00,
   };
   /* applegpu-nt 32023.917 depth2d_array::sample_compare: discarded r3
    * publishes the dynamic layer on slot 1. */
   const uint8_t native_shadow_array[] = {
      0x17, 0x05, 0x54, 0x06, 0x00, 0x0c,
      0x00, 0x00, 0xf0, 0xc0, 0x04, 0x00,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   CHECK(agx3_tex_stage_array_layer(AGX_ISA_ENCODING_AGX3B, 4, 1, 0,
                                    false, true, bytes, &len));
   CHECK(len == sizeof(native_slot1) && !memcmp(bytes, native_slot1, len));
   CHECK(memcmp(bytes,
                (const uint8_t[]){0x17,0x05,0x54,0x06,0,0x08,0,4,
                                  0xf0,0xc0,0x08,0}, len));
   CHECK(agx3_tex_stage_array_layer(AGX_ISA_ENCODING_AGX3B, 12, 6, 0,
                                    false, true, bytes, &len));
   CHECK(len == sizeof(native_register) &&
         !memcmp(bytes, native_register, len));
   CHECK(agx3_tex_stage_array_layer(AGX_ISA_ENCODING_AGX3B, 6, 1, 0,
                                    false, true, bytes, &len));
   CHECK(len == sizeof(native_shadow_array) &&
         !memcmp(bytes, native_shadow_array, len));
   return 0;
}

static int
test_cube_coordinate_stage(void)
{
   /* Exact own-MSL texturecube CEM sequence and face publication. */
   static const uint8_t cem[3][12] = {
      {0x17,0x01,0x54,0x00,0x03,0x04,0x0d,0x52,0x54,0x21,0x92,0x00},
      {0x17,0x01,0x54,0x07,0x03,0x04,0x0d,0x52,0x54,0x21,0x92,0x04},
      {0x17,0x01,0x54,0x02,0x03,0x04,0x0c,0x50,0x50,0x2f,0x92,0x08},
   };
   static const uint8_t rcp[10] = {
      0xaf,0x00,0x54,0x00,0x03,0x00,0x10,0x48,0x20,0x00,
   };
   static const uint8_t native[] = {
      0x17,0x05,0x54,0x06,0x00,0x04,
      0x00,0x04,0xf0,0xc0,0x00,0x00,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   for (unsigned axis = 0; axis < 3; ++axis) {
      CHECK(agx3_tex_cube_cem_calc(AGX_ISA_ENCODING_AGX3B, 4, 6, 20,
                                   axis, axis == 2, bytes, &len));
      CHECK(len == sizeof(cem[axis]) && !memcmp(bytes, cem[axis], len));
   }
   CHECK(agx3_tex_cube_cem_rcp(AGX_ISA_ENCODING_AGX3B, bytes, &len));
   CHECK(len == sizeof(rcp) && !memcmp(bytes, rcp, len));
   static const uint8_t finalize[2][12] = {
      {0x09,0x0f,0x2e,0x81,0x23,0xa0,0x42,0x00,0x00,0x02,0x02,0x00},
      {0x19,0x05,0x3e,0x01,0x23,0xa0,0x42,0x00,0x00,0x02,0x62,0x00},
   };
   for (unsigned component = 0; component < 2; ++component) {
      CHECK(agx3_tex_cube_cem_finalize(AGX_ISA_ENCODING_AGX3B, component,
                                       component == 1, bytes, &len));
      CHECK(len == sizeof(finalize[component]) &&
            !memcmp(bytes, finalize[component], len));
   }

   /* Fresh dynamic argument-buffer cube witness. CEM uses allocator-owned
    * halves8/16/12, its reciprocal is in-place, and both finalizers join the
    * descriptor/face transaction on slot6. */
   static const uint8_t dynamic_cem[3][12] = {
      {0x17,0x01,0x54,0x04,0x03,0x0c,0x1d,0x5a,0x54,0x21,0x92,0x00},
      {0x17,0x01,0x54,0x08,0x03,0x0c,0x1d,0x5a,0x54,0x21,0x92,0x04},
      {0x17,0x01,0x54,0x06,0x03,0x0c,0x1c,0x58,0x50,0x2f,0x92,0x08},
   };
   const uint16_t dynamic_dest[3] = {8, 16, 12};
   for (unsigned axis = 0; axis < 3; ++axis) {
      CHECK(agx3_tex_cube_cem_calc_to(
         AGX_ISA_ENCODING_AGX3B, 12, 14, 22, dynamic_dest[axis], axis,
         axis == 2, bytes, &len));
      CHECK(len == sizeof(dynamic_cem[axis]) &&
            !memcmp(bytes, dynamic_cem[axis], len));
   }
   static const uint8_t dynamic_rcp[10] = {
      0xaf,0x00,0x54,0x08,0x03,0x10,0x10,0x48,0x20,0x00,
   };
   CHECK(agx3_tex_cube_cem_rcp_to(AGX_ISA_ENCODING_AGX3B, 8, bytes, &len));
   CHECK(len == sizeof(dynamic_rcp) && !memcmp(bytes, dynamic_rcp, len));
   static const uint8_t dynamic_finalize[2][12] = {
      {0x09,0x11,0x2e,0x89,0x23,0xa0,0x42,0x00,0x00,0x16,0x02,0x00},
      {0x19,0x0d,0x3e,0x09,0x23,0xa0,0x42,0x00,0x00,0x16,0x02,0x00},
   };
   for (unsigned component = 0; component < 2; ++component) {
      CHECK(agx3_tex_cube_cem_finalize_to(
         AGX_ISA_ENCODING_AGX3B, component, 8,
         dynamic_dest[component + 1], 6, bytes, &len));
      CHECK(len == sizeof(dynamic_finalize[component]) &&
            !memcmp(bytes, dynamic_finalize[component], len));
   }
   CHECK(agx3_tex_stage_cube_z(AGX_ISA_ENCODING_AGX3B, 2, 1, 0,
                               false, true, bytes, &len));
   CHECK(len == sizeof(native) && !memcmp(bytes, native, len));
   /* applegpu-nt 32023.917 depthcube_array::sample_compare combines CEM's
    * face and discarded 32-bit layer r3 in one distinct PACK_SMPIR. */
   static const uint8_t native_shadow_array[] = {
      0x17,0x05,0x54,0x06,0x00,0x06,
      0x08,0x00,0xf0,0xc0,0x29,0x00,
   };
   CHECK(agx3_tex_stage_cube_array_shadow(
      AGX_ISA_ENCODING_AGX3B, 6, 1, 0, false, true, bytes, &len));
   CHECK(len == sizeof(native_shadow_array) &&
         !memcmp(bytes, native_shadow_array, len));
   CHECK(!agx3_tex_stage_cube_z(AGX_ISA_ENCODING_AGX3C, 2, 1, 0,
                                false, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_z(AGX_ISA_ENCODING_AGX3B, 3, 1, 0,
                                false, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_z(AGX_ISA_ENCODING_AGX3B, 2, 1, 0,
                                true, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_array_shadow(
      AGX_ISA_ENCODING_AGX3C, 6, 1, 0, false, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_array_shadow(
      AGX_ISA_ENCODING_AGX3B, 7, 1, 0, false, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_array_shadow(
      AGX_ISA_ENCODING_AGX3B, 6, 1, 0, true, true, bytes, &len));
   CHECK(!agx3_tex_cube_cem_calc(AGX_ISA_ENCODING_AGX3C, 4, 6, 20, 0,
                                 false, bytes, &len));
   CHECK(!agx3_tex_cube_cem_calc(AGX_ISA_ENCODING_AGX3B, 4, 6, 20, 0,
                                 true, bytes, &len));
   CHECK(!agx3_tex_cube_cem_calc(AGX_ISA_ENCODING_AGX3B, 128, 6, 20, 0,
                                 false, bytes, &len));
   CHECK(!agx3_tex_cube_cem_finalize(AGX_ISA_ENCODING_AGX3C, 0, false,
                                     bytes, &len));
   CHECK(!agx3_tex_cube_cem_finalize(AGX_ISA_ENCODING_AGX3B, 0, true,
                                     bytes, &len));
   CHECK(!agx3_tex_cube_cem_finalize(AGX_ISA_ENCODING_AGX3B, 2, false,
                                     bytes, &len));
   return 0;
}

static int
test_multisample_fetch(void)
{
   /* A18 Pro own-MSL texture2d_ms reads, isolated for samples 0..3. */
   static const uint8_t native[4][14] = {
      {0x05,0x80,0x0c,0xb8,0xb0,0x00,0x80,0x14,0,0,0,0,1,0},
      {0x05,0x80,0x0c,0xb8,0xb0,0x02,0x80,0x14,0,0,0,0,1,0},
      {0x05,0x80,0x0c,0xb8,0xb0,0x04,0x80,0x14,0,0,0,0,1,0},
      {0x05,0x80,0x0c,0xb8,0xb0,0x06,0x80,0x14,0,0,0,0,1,0},
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   struct agx3_tex_fetch_desc d = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .slot = 1, .wait_mask = 1, .coord_kill = true, .final = true,
      .multisample = true,
   };

   for (d.sample = 0; d.sample < 4; ++d.sample) {
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
      CHECK(len == sizeof(native[d.sample]));
      CHECK(!memcmp(bytes, native[d.sample], len));
   }

   d.sample = 4;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.sample = 0;
   d.coord = 16;
   d.array = true;
   d.texture = 1;
   d.wait_mask = 2;
   static const uint8_t native_ms_array[14] = {
      0x05,0x80,0x0c,0xb9,0x30,0x01,0xd9,0x11,0x80,0,0,0,1,0,
   };
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   CHECK(len == sizeof(native_ms_array) &&
         !memcmp(bytes, native_ms_array, len));
   d.sample = 1;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.sample = 0;
   d.coord = 0;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.array = false;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, bytes, &len));
   d.multisample = false;
   d.sample = 1;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.sample = 0;
   d.multisample = true;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_value = 2;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   return 0;
}

static int
test_multisample_sample_stage(void)
{
   /* Own-MSL buffer-indexed sample: discarded r2h is published before the
    * otherwise sample-0 texture2d_ms read. */
   static const uint8_t native[] = {
      0x17,0x05,0x54,0x04,0x00,0x00,
      0x00,0x00,0xd0,0xc0,0x08,0x00,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   CHECK(agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 4, 0,
                                  false, true, bytes, &len));
   CHECK(len == sizeof(native) && !memcmp(bytes, native, len));
   CHECK(!agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 256, 0,
                                   false, false, bytes, &len));
   CHECK(!agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3C, 4, 0,
                                   false, true, bytes, &len));
   CHECK(!agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 4, 0x40,
                                   false, true, bytes, &len));
   CHECK(!agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 4, 0,
                                   true, true, bytes, &len));
   return 0;
}

int
main(void)
{
   if (test_register_lod() || test_fetch_lod() ||
       test_register_array_layer() || test_cube_coordinate_stage() ||
       test_multisample_fetch() ||
       test_multisample_sample_stage() ||
       test_staging_source_lifetime() || test_exact_witnesses() ||
       test_supported_axes() || test_fail_closed() ||
       test_sample_witnesses() || test_copy_witnesses() ||
       test_sample_fail_closed() || test_array_fetch() ||
       test_bindless_operand_kinds() || test_indexed_sample_witnesses() ||
       test_sample_mask_domain() || test_indexed_fetch_native() ||
       test_gradient_and_register_bias())
      return 1;
   printf("agx3 texture core: %u checks passed\n", checks);
   return 0;
}
