/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_tex_core.h"

#include "agx3_alu_core.h"
#include "agx3_binder.h"

#include <string.h>

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

static bool
one_or_zero(uint16_t value)
{
   return (value & (value - 1)) == 0;
}

static bool
fetch_ok(enum agx_isa_encoding enc, const struct agx3_tex_fetch_desc *d)
{
   if (d == NULL || (enc != AGX_ISA_ENCODING_AGX3B &&
                     enc != AGX_ISA_ENCODING_AGX3C))
      return false;

   if ((d->width != AGX3_TEX_W16 && d->width != AGX3_TEX_W32) ||
       d->dest >= 128 || (d->width == AGX3_TEX_W32 && (d->dest & 1)) ||
       (d->coord & 3) || d->coord > 60 ||
       (enc == AGX_ISA_ENCODING_AGX3B ? (!d->mask || d->mask > 15)
                                    : (d->mask != 1 && d->mask != 15)) ||
       d->texture > 15)
      return false;

   /* A18 Pro witnesses both 2D-MS and 2D-MS-array reads.  The non-array
    * form uses staging halves 0/2 and carries a bounded immediate sample.
    * The array form uses halves 16/18 and PACK_SMPIR publishes the packed
    * sample/layer word separately. */
   if ((d->sample && !d->multisample) ||
       (d->multisample &&
        (enc != AGX_ISA_ENCODING_AGX3B || d->sample > 3 ||
         (d->array && d->sample) || d->coord != (d->array ? 16 : 0))))
      return false;

   /* Own-MSL texture3d::read witnesses use f76=3 with a third integer
    * coordinate in the contiguous staging bank. The immediate and indirect
    * texture forms are both captured on G17P; no AGX3c or combined
    * array/multisample form is grounded. */
   if (d->three_d &&
       (enc != AGX_ISA_ENCODING_AGX3B || d->array || d->multisample ||
        d->register_lod))
      return false;

   const unsigned tk = d->texture_kind ?: AGX3_TEX_OPERAND_IMMEDIATE;
   const unsigned sk = d->sampler_kind ?: AGX3_TEX_OPERAND_IMMEDIATE;
   if (tk > AGX3_TEX_OPERAND_UNIFORM || sk > AGX3_TEX_OPERAND_UNIFORM ||
       (tk != AGX3_TEX_OPERAND_IMMEDIATE &&
        ((d->texture_value & 1) || d->texture_value > 252)) ||
       (sk == AGX3_TEX_OPERAND_IMMEDIATE ? d->sampler_value > 15
         : ((d->sampler_value & 1) || d->sampler_value > 252)) ||
       (d->texture_kill && tk != AGX3_TEX_OPERAND_REGISTER) ||
       (d->multisample &&
        (sk != AGX3_TEX_OPERAND_IMMEDIATE || d->sampler_value != 0)))
      return false;
   if (d->three_d &&
       (sk != AGX3_TEX_OPERAND_IMMEDIATE || d->sampler_value != 0))
      return false;

   /* The level. f29 is 12 bits of 1/64 fixed point; the per-lane form has no
    * immediate at all, and is witnessed on AGX3b only. */
   if (d->lod_x64 > 2047 || (d->register_lod && d->lod_x64) ||
       (d->register_lod && enc != AGX_ISA_ENCODING_AGX3B) ||
       (d->multisample && (d->lod_x64 || d->register_lod)))
      return false;

   const unsigned slots = enc == AGX_ISA_ENCODING_AGX3C ? 8 : 6;
   const uint16_t waits = enc == AGX_ISA_ENCODING_AGX3C ? 0x7f : 0x3f;
   return d->slot >= 1 && d->slot <= slots &&
          (d->wait_mask & ~waits) == 0 && one_or_zero(d->wait_mask);
}

bool
agx3_tex_encode_fetch(enum agx_isa_encoding enc,
                      const struct agx3_tex_fetch_desc *d,
                      uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (!fetch_ok(enc, d))
      return false;

   struct agx3_field_kv base[20];
   unsigned nb = 0;
   base[nb++] = (struct agx3_field_kv)KV(9, 2);
   base[nb++] = (struct agx3_field_kv)KV(23, 5);
   base[nb++] = (struct agx3_field_kv)KV(38, 1);
   base[nb++] = (struct agx3_field_kv)KV(52, 1);
   base[nb++] = (struct agx3_field_kv)KV(89, 3);
   base[nb++] = (struct agx3_field_kv)KV(90, 3);
   const enum agx3_tex_operand_kind tk =
      d->texture_kind ? d->texture_kind : AGX3_TEX_OPERAND_IMMEDIATE;
   const enum agx3_tex_operand_kind sk =
      d->sampler_kind ? d->sampler_kind : AGX3_TEX_OPERAND_IMMEDIATE;
   const bool immediate_operands = tk == AGX3_TEX_OPERAND_IMMEDIATE &&
                                   sk == AGX3_TEX_OPERAND_IMMEDIATE;

   base[nb++] = (struct agx3_field_kv)KV(103, tk);
   base[nb++] = (struct agx3_field_kv)KV(104, tk == AGX3_TEX_OPERAND_IMMEDIATE ? 1 : 2);
   base[nb++] = (struct agx3_field_kv)KV(112, 4);
   base[nb++] = (struct agx3_field_kv)KV(118, 3);
   base[nb++] = (struct agx3_field_kv)KV(119, 3);
   base[nb++] = (struct agx3_field_kv)KV(132, sk);
   base[nb++] = (struct agx3_field_kv)KV(133, sk == AGX3_TEX_OPERAND_IMMEDIATE ? 1 : 2);
   base[nb++] = (struct agx3_field_kv)KV(138, 1);
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(24, 2);
      base[nb++] = (struct agx3_field_kv)KV(60, 2);
      base[nb++] = (struct agx3_field_kv)KV(61, 164);
   }

   struct agx3_field_kv ov[24];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(10, d->width);
   ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(19, d->coord);
   ov[n++] = (struct agx3_field_kv)KV(26, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(27, d->coord_kill ? 1 : 0);
   /* MS-array names its packed third coordinate with f29=2. Non-MS fetches
    * use f29 for immediate LOD and f37 to mark that operand. */
   if (d->multisample && d->array) {
      ov[n++] = (struct agx3_field_kv)KV(29, 2);
      ov[n++] = (struct agx3_field_kv)KV(37, 1);
   } else if (!d->register_lod) {
      ov[n++] = (struct agx3_field_kv)KV(37, d->multisample ? 0 : 1);
      if (d->lod_x64)
         ov[n++] = (struct agx3_field_kv)KV(29, d->lod_x64);
   }
   ov[n++] = (struct agx3_field_kv)KV(75, d->mask);
   ov[n++] = (struct agx3_field_kv)KV(
      76, d->three_d ? 3 : (d->multisample ? 2 : 1));
   if (d->array)
      ov[n++] = (struct agx3_field_kv)KV(
         77, d->multisample ? 1 : 2); /* MS array vs register-layer array */
   ov[n++] = (struct agx3_field_kv)KV(78, 1);  /* texel fetch */
   /* Each operand lands in the field its kind selects: 95/96/97 for the
    * texture, 124/125/126 for the sampler. */
   ov[n++] = (struct agx3_field_kv)KV(
      tk == AGX3_TEX_OPERAND_IMMEDIATE   ? 95
      : tk == AGX3_TEX_OPERAND_REGISTER  ? 96
                                         : 97,
      tk == AGX3_TEX_OPERAND_IMMEDIATE ? d->texture : d->texture_value);
   ov[n++] = (struct agx3_field_kv)KV(
      sk == AGX3_TEX_OPERAND_IMMEDIATE   ? 124
      : sk == AGX3_TEX_OPERAND_REGISTER  ? 125
                                         : 126,
      d->sampler_value);
   ov[n++] = (struct agx3_field_kv)KV(107, d->texture_kill);
   ov[n++] = (struct agx3_field_kv)KV(
      139, d->multisample ? 0 : d->register_lod ? 4 : 2);
   if (enc == AGX_ISA_ENCODING_AGX3B)
      ov[n++] = (struct agx3_field_kv)KV(147, d->final ? 2 : 0);
   else {
      ov[n++] = (struct agx3_field_kv)KV(150, d->final ? 2 : 0);
      ov[n++] = (struct agx3_field_kv)KV(155, 1);
   }
   for (unsigned i = 0; i < 8; ++i)
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(62 + i, 1);

   /*
    * Immediate operands are ep3; register and uniform operands are ep4. Only
    * AGX3b is witnessed for the non-immediate kinds, so AGX3c refuses them
    * rather than assuming its encoding points line up.
    */
   if (!immediate_operands && enc != AGX_ISA_ENCODING_AGX3B)
      return false;
   const int ep =
      enc == AGX_ISA_ENCODING_AGX3B ? (immediate_operands ? 3 : 4) : 0;
   if (!agx3_bind_compose(enc, "smp", ep, base, nb, ov, n, out, len)) {
      memset(out, 0, AGX3_TEX_MAX_LENGTH);
      *len = 0;
      return false;
   }

   if (d->multisample && !d->array) {
      /* Own-MSL reads of samples 0..3 isolate this two-bit immediate: byte 5
       * is 00/02/04/06 while the remaining sample-0 instruction is fixed. */
      if (out[5] & 0x06) {
         memset(out, 0, AGX3_TEX_MAX_LENGTH);
         *len = 0;
         return false;
      }
      out[5] |= d->sample << 1;
   }
   return true;
}

static bool
mask_witnessed(uint8_t mask)
{
   /* All 15 nonempty RGBA subsets are witnessed by native fragment samples.
    * Results remain densely packed in selected-component order. */
   return mask != 0 && (mask & ~15u) == 0;
}

static bool
sample_ok(enum agx_isa_encoding enc, const struct agx3_tex_sample_desc *d)
{
   if (d == NULL || (enc != AGX_ISA_ENCODING_AGX3B &&
                     enc != AGX_ISA_ENCODING_AGX3C))
      return false;

   if (d->sample_abi > AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT ||
       (d->sample_abi == AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT &&
        enc != AGX_ISA_ENCODING_AGX3B))
      return false;

   if ((d->sample_group_nonterminal || d->sample_group_end) &&
       (enc != AGX_ISA_ENCODING_AGX3B ||
        d->lod == AGX3_TEX_LOD_GRADIENT ||
        (d->sample_group_nonterminal && d->sample_group_end)))
      return false;

   if ((d->width != AGX3_TEX_W16 && d->width != AGX3_TEX_W32) ||
       d->dest >= 128 || (d->width == AGX3_TEX_W32 && (d->dest & 1)) ||
       (d->coord & 3) || d->coord > 60 || !mask_witnessed(d->mask) ||
       d->texture > 15 || d->sampler > 15)
      return false;

   if (d->lod == AGX3_TEX_LOD_LEVEL) {
      /* 12-bit immediate level field; witnessed up to 464 (level 7.25). */
      if (d->lod_x64 > 0xfff)
         return false;
   } else if (d->lod == AGX3_TEX_LOD_BIAS) {
      /* The same 12-bit x64 field, but read as *signed*: Apple encodes a
       * negative bias in two's complement (bias(-1.0) is 4032, bias(-32.0)
       * is 2048) and saturates at 2047 / 2048. The caller passes the field
       * already masked to 12 bits. Apple emits the plain implicit form
       * rather than a zero bias, so refuse the degenerate encoding. */
      if (enc != AGX_ISA_ENCODING_AGX3B || d->lod_x64 == 0 ||
          d->lod_x64 > 0xfff)
         return false;
   } else if (d->lod == AGX3_TEX_LOD_REGISTER ||
              d->lod == AGX3_TEX_LOD_BIAS_REGISTER ||
              d->lod == AGX3_TEX_LOD_GRADIENT) {
      /* The per-lane level, the per-lane bias and the four gradient scalars
       * are all witnessed on AGX3b only, and none of them carries the
       * immediate operand. */
      if (enc != AGX_ISA_ENCODING_AGX3B || d->lod_x64 != 0)
         return false;
   } else if (d->lod != AGX3_TEX_LOD_AUTO || d->lod_x64 != 0) {
      return false;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && d->mask != 1 && d->mask != 3 &&
       d->mask != 8 && d->mask != 12 && d->mask != 15)
      return false;

   /* G17P names a dynamic array layer directly with f31/f37 in every
    * witnessed LOD mode. Do not accept a value the 32-bit register operand
    * cannot name, or silently ignore a layer on a non-array sample. */
   if ((d->array &&
        (enc != AGX_ISA_ENCODING_AGX3B || (d->array_layer & 1) ||
         d->array_layer > 254)) ||
       (!d->array && d->array_layer != 0))
      return false;

   /* The 3D dimension is f76 = 3 and is orthogonal to the LOD shape exactly
    * as the 2D one is. Five texture3d fragments, differing only in the LOD
    * argument, differ only in f139 and its operand:
    *
    *   t.sample(s, uvw)                 05800cb8b0008001010010001200
    *   t.sample(s, uvw, bias(0.5f))     05800cb8b0003b00010010021200  f139 1, f29 32
    *   t.sample(s, uvw, bias(in.dx.x))  05800cb8b0009001010010001200  f139 3
    *   t.sample(s, uvw, level(in.dx.x)) 05800cb8b0008821010010001200  f139 4
    *   t.sample(s, uvw, gradient3d(..)) 05800cb8b0009c01000010001200  f139 5
    *
    * and the per-lane staging is the same bank-6 publication (f5 = 6) our 2D
    * path already emits, read straight out of the level(reg) kernel.
    *
    * gradient3d carries *six* scalars rather than four; the extra pair
    * continues the same interleave into banks 16 and 18, witnessed on two
    * literal-gradient kernels (see AGX3_TEX_GRADIENT_BANK_DDX_Z). Array,
    * gather and comparison combinations remain closed. */
   if (d->three_d &&
       (enc != AGX_ISA_ENCODING_AGX3B || d->array || d->cube ||
        d->gather != AGX3_TEX_GATHER_NONE || d->shadow))
      return false;

   /* Native cube sampling is a distinct dimension (SMP f76=4). Apple
    * publishes its third f32 coordinate with PACK_SMPIR f74=10, then samples
    * fixed coordinate bank 0. Implicit LOD and the ordinary immediate and
    * register explicit-LOD selectors are independently witnessed. Cube and
    * cube-array comparisons reuse that CEM path with their own packed face /
    * layer operands. */
   if (d->cube &&
       (enc != AGX_ISA_ENCODING_AGX3B || d->three_d ||
        d->coord != 0 ||
        (d->lod != AGX3_TEX_LOD_AUTO && d->lod != AGX3_TEX_LOD_LEVEL &&
         d->lod != AGX3_TEX_LOD_REGISTER) ||
        d->gather != AGX3_TEX_GATHER_NONE))
      return false;

   const unsigned tk = d->texture_kind ? d->texture_kind : 1;
   const unsigned sk = d->sampler_kind ? d->sampler_kind : 1;
   if (tk > 3 || sk > 3 ||
       ((tk != 1 || sk != 1) && enc != AGX_ISA_ENCODING_AGX3B) ||
       (tk != 1 && ((d->texture_value & 1) || d->texture_value > 252)) ||
       (sk != 1 && ((d->sampler_value & 1) || d->sampler_value > 252)) ||
       (d->texture_kill && tk != 2) || (d->sampler_kill && sk != 2))
      return false;

   /*
    * A dimensioned comparison used to require fixed texture and sampler
    * state. The descriptor operand class is orthogonal to both the depth
    * comparison and the packed array layer, and Apple's own bytes say so:
    * an `array<depth2d_array<float>, 8>` sampled through a runtime index,
    * against the same shader with a fixed texture and sampler --
    *
    *   fixed     1d800ca090002004010000001200          ep3, 14 B
    *   indexed   1dc00ca09c002004930380011200802000000000   ep4, 20 B
    *
    * differ in exactly the four descriptor fields and the endpoint they
    * imply: the texture gains f97 = 24 with f103/f104 going 1 -> 3/2, the
    * sampler gains f126 = 28 with f132/f133 going 1 -> 3/2. **f143 (the
    * comparison) and f77 = 2 (the packed layer) are identical in both**, so
    * neither depends on where the descriptor came from.
    *
    * That distinction is what a real application runs into. A shader with a
    * single texture instruction gets immediate descriptors from
    * agx_nir_lower_g17p_combined_sampler(); one with several -- SuperTuxKart's
    * sunlightshadow.frag samples its cascade four times -- falls back to
    * agx_nir_lower_g17p_indexed_samplers() and arrives here with registers,
    * where the whole shader used to fail closed.
    *
    * The non-driver sample ABIs stay refused: those change the operand
    * profile itself rather than the descriptor's class.
    */
   if (d->shadow && (d->array || d->cube) &&
       d->sample_abi != AGX3_TEX_SAMPLE_ABI_DRIVER)
      return false;

   /* Cube operand pairs are witnessed as immediate/immediate,
    * uniform/uniform and register/register. */
   if (d->cube && tk != sk)
      return false;

   /* The coordinate offset is one immediate nibble pair in f43 plus two
    * selectors, f51 and f141, and it is orthogonal to every other property
    * of the sample. Each of these differs from its offset-free control by
    * exactly {f43 = (y << 4) | x, f51 = 1, f141 = 1} and nothing else
    * (metal-nt 27.1, -arch applegpu_g17p, fragment render pipelines):
    *
    *   t.sample(s, uv)                          05800cb8b0000000010010001200
    *   t.sample(s, uv, int2(1,1))               05800cb8b0000102018810001200
    *   t.sample(s, uv, level(2), int2(1,1))     05800cb8b0001b02018810081200
    *   t.sample(s, uv, level(reg), int2(1,1))   05800cb8b0001102018810001200
    *   t.sample(s, uv, bias(reg), int2(1,1))    05800cb8b0008102018810001200
    *   t.sample(s, uv, grad, int2(1,1))         05800cb8b0000902008810001200
    *   t.gather(s, uv, int2(1,1))               05800ca490000102018800001200
    *   t.sample(s, uv, layer, int2(1,1))        05800cb8b0001bb7018810001200
    *   depth.sample_compare(s, uv, r, int2(1,1))1d800ca090002102018800001200
    *   heap t.sample(s, uv, int2(1,1))          05a00cb8b8000102938b90011200...
    *
    * and the value sweep is the same in every LOD mode: offset (1,1) is
    * f43 = 17, (2,0) is f43 = 2, (0,2) is f43 = 32.
    *
    * Offset (0, 0) is folded to the plain form by native, so do not
    * manufacture the enabled encoding without a corresponding witness.
    *
    * f43 is a single shared immediate slot, and an *array* sample carrying
    * an immediate level or bias already claims it (see the LOD block below).
    * Apple does not co-encode an offset there: `t.sample(s, uv, 3u,
    * bias(0.5f), int2(1,1))` is 05800cb8b000451b010010001202, which keeps
    * the bias in f43 and selects f141 = 2 -- a form that publishes the
    * offset through an extra twelve-byte staged operand. That staging has no
    * slot plan here, so the collision is refused rather than silently
    * dropping one of the two values.
    *
    * 3D and cube offsets are not witnessed and stay closed, and so is a
    * register descriptor pair: the immediate and uniform pairs above each
    * take the offset orthogonally, but the register pair has no offset
    * witness of its own.
    */
   const bool array_immediate_lod =
      d->array &&
      (d->lod == AGX3_TEX_LOD_BIAS || d->lod == AGX3_TEX_LOD_LEVEL);
   if (d->offset &&
       (enc != AGX_ISA_ENCODING_AGX3B || d->offset_xy == 0 ||
        d->three_d || d->cube || array_immediate_lod ||
        tk == AGX3_TEX_OPERAND_REGISTER || sk == AGX3_TEX_OPERAND_REGISTER))
      return false;

   if (!d->offset && d->offset_xy != 0)
      return false;

   /* Metal's gather takes no LOD argument, so every gather witness is an
    * implicit-LOD one on AGX3b. The array layer, the coordinate offset and
    * the depth comparison do compose with it, each contributing exactly the
    * fields it contributes to an ordinary sample:
    *
    *   t.gather(s, uv)                       05800ca490000000010000001200
    *   t.gather(s, uv, layer)                05800ca490000db4010000001200  f77/f31/f37
    *   t.gather(s, uv, layer, int2(1,1))     05800ca490001bb7018800001200  + f43/f51/f141
    *   depth.gather_compare(s, uv, r)        05800ca490002000010000001200  + f143
    *   depth.gather_compare(s, uv, r, off)   05800ca490002102018800001200  + f43/f51/f141
    *
    * 3D and cube gathers are not witnessed and stay closed.
    */
   if (d->gather != AGX3_TEX_GATHER_NONE &&
       (d->gather > AGX3_TEX_GATHER_A || enc != AGX_ISA_ENCODING_AGX3B ||
        d->lod != AGX3_TEX_LOD_AUTO || d->three_d || d->cube))
      return false;

   /* Apple's depth sample_compare witnesses stage their reference value into
    * fixed bank 4 and set f143. The comparison is orthogonal to the LOD
    * shape: each of these differs from the corresponding *colour* sample by
    * f143 = 1, f75 = 1 instead of 15, and the dropped f74 mask-group bit,
    * and by nothing else.
    *
    *   depth.sample_compare(s, uv, r)             1d800ca090002000010000001200
    *   depth.sample_compare(s, uv, r, level(0))   1d800ca090002900010000001200  f139=2
    *   depth.sample_compare(s, uv, r, level(2))   1d800ca090002900010000081200  f139=2 f29=128
    *   depth.sample_compare(s, uv, r, bias(0.5))  1d800ca090002700010000021200  f139=1 f29=32
    *   depth.sample_compare(s, uv, r, level(reg)) 1d800ca090002001010000001200  f139=4
    *   depth.sample_compare(s, uv, r, bias(reg))  1d800ca090002400010000001200  f139=3
    *   depth.sample_compare(s, uv, r, grad)       1d800ca090002401000000001200  f139=5
    *   depth.gather_compare(s, uv, r)             05800ca490002000010000001200  f144=1
    *
    * A comparison gather returns four texels, so it carries the full f75
    * mask the ordinary gather does; every other comparison returns one.
    *
    * The array and cube comparisons here use their own separately packed
    * layer / CEM operands (f77 = 2, f76 = 4) rather than the register-layer
    * form the LOD sweep above was taken on, so they keep the implicit LOD
    * they were witnessed with. */
   if (d->shadow &&
       (enc != AGX_ISA_ENCODING_AGX3B || d->three_d ||
        d->mask != (d->gather != AGX3_TEX_GATHER_NONE ? 15u : 1u) ||
        ((d->array || d->cube) && d->lod != AGX3_TEX_LOD_AUTO)))
      return false;

   const unsigned slots = enc == AGX_ISA_ENCODING_AGX3C ? 8 : 6;
   const uint16_t waits = enc == AGX_ISA_ENCODING_AGX3C ? 0xff : 0x3f;
   return d->slot >= 1 && d->slot <= slots &&
          (d->wait_mask & ~waits) == 0 && one_or_zero(d->wait_mask);
}

bool
agx3_tex_encode_sample(enum agx_isa_encoding enc,
                       const struct agx3_tex_sample_desc *d,
                       uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (!sample_ok(enc, d))
      return false;

   /* The fetch baseline plus f74=1: the sample witnesses carry the extra
    * mask-group bit on every mask value, where the fetch form does not.
    * A gather is the one sample shape that also clears it. */
   const unsigned tk = d->texture_kind ? d->texture_kind : 1;
   const unsigned sk = d->sampler_kind ? d->sampler_kind : 1;
   const bool immediate = tk == 1 && sk == 1;
   const bool apple_fragment_abi =
      d->sample_abi == AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT;
   const bool shadow_array = d->shadow && d->array;
   unsigned operand_profile = apple_fragment_abi ? 3 : 0;
   struct agx3_field_kv base[20];
   unsigned nb = 0;
   base[nb++] = (struct agx3_field_kv)KV(9, 2);
   base[nb++] = (struct agx3_field_kv)KV(23, 5);
   base[nb++] = (struct agx3_field_kv)KV(38, 1);
   base[nb++] = (struct agx3_field_kv)KV(52, 1);
   if (d->gather == AGX3_TEX_GATHER_NONE && !d->shadow)
      base[nb++] = (struct agx3_field_kv)KV(74, 1);
   base[nb++] = (struct agx3_field_kv)KV(89, 3);
   base[nb++] = (struct agx3_field_kv)KV(90, 3);
   base[nb++] = (struct agx3_field_kv)KV(103, tk);
   base[nb++] =
      (struct agx3_field_kv)KV(104, tk == AGX3_TEX_OPERAND_IMMEDIATE ? 1 : 2);
   /* The driver descriptor ABI uses f83/f112=0/4 for every dimension.
    * Apple's f83/f112=4/8 fragment profile is a complete, incompatible ABI,
    * selected only by AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT.  In particular,
    * using Apple's profile with the driver's 2D-array descriptor produces a
    * fixed near-half-gray sample on G17P, just as it does for cubes.
    */
   if (operand_profile & 1)
      base[nb++] = (struct agx3_field_kv)KV(83, 4);
   base[nb++] = (struct agx3_field_kv)KV(
      112, (operand_profile & 2) ? 8 : 4);
   base[nb++] = (struct agx3_field_kv)KV(118, 3);
   base[nb++] = (struct agx3_field_kv)KV(119, 3);
   base[nb++] = (struct agx3_field_kv)KV(132, sk);
   base[nb++] =
      (struct agx3_field_kv)KV(133, sk == AGX3_TEX_OPERAND_IMMEDIATE ? 1 : 2);
   base[nb++] = (struct agx3_field_kv)KV(138, 1);
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(24, 2);
      base[nb++] = (struct agx3_field_kv)KV(60, 2);
      base[nb++] = (struct agx3_field_kv)KV(61, 164);
   }

   struct agx3_field_kv ov[33];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(10, d->width);
   ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(19, d->coord);
   ov[n++] = (struct agx3_field_kv)KV(26, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(27, d->coord_kill ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(75, d->mask);
   ov[n++] = (struct agx3_field_kv)KV(76, d->cube ? 4 : d->three_d ? 3 : 1);
   ov[n++] = (struct agx3_field_kv)KV(78, 2);  /* filtered sample */

   /* f156 is a group terminator, not a per-instruction implicit-LOD bit.
    * A lone AUTO/BIAS, staged register LOD, array, or cube sample forms a
    * one-op group and keeps the historical encoding. The stream packer marks
    * nonterminal members to suppress that default and supplies
    * sample_group_end when a multi-op group ends on a shape (including an
    * immediate LEVEL) that would not infer it in isolation. Explicit-gradient
    * samples are outside these groups. */
   const bool intrinsic_group =
      d->lod == AGX3_TEX_LOD_AUTO || d->lod == AGX3_TEX_LOD_BIAS ||
      d->lod == AGX3_TEX_LOD_REGISTER || d->lod == AGX3_TEX_LOD_BIAS_REGISTER ||
      (d->lod != AGX3_TEX_LOD_GRADIENT && (d->array || d->cube));
   if (d->sample_group_end ||
       (intrinsic_group && !d->sample_group_nonterminal))
      ov[n++] = (struct agx3_field_kv)KV(156, 1);
   if (d->array) {
      /* Honeykrisp's driver descriptor ABI pairs noncube arrays with the
       * sampler-private layer staged by PACK_SMPIR.  Native Apple fragment
       * witnesses use the direct f31/f37 path as part of their complete 4/8
       * ABI; cube arrays retain their independently witnessed direct path.
       */
      const bool staged_driver_array =
         !d->cube && d->lod == AGX3_TEX_LOD_AUTO &&
         d->sample_abi == AGX3_TEX_SAMPLE_ABI_DRIVER;
      if (shadow_array || staged_driver_array) {
         ov[n++] = (struct agx3_field_kv)KV(77, 2);
      } else {
         ov[n++] = (struct agx3_field_kv)KV(31, d->array_layer);
         ov[n++] = (struct agx3_field_kv)KV(37, 3);
         ov[n++] = (struct agx3_field_kv)KV(77, 1);
      }
   }
   ov[n++] = (struct agx3_field_kv)KV(94 + tk, tk == 1 ? d->texture : d->texture_value);
   ov[n++] = (struct agx3_field_kv)KV(123 + sk, sk == 1 ? d->sampler : d->sampler_value);
   if (d->texture_kill)
      ov[n++] = (struct agx3_field_kv)KV(107, 1);
   if (d->sampler_kill)
      ov[n++] = (struct agx3_field_kv)KV(136, 1);
   if (d->lod == AGX3_TEX_LOD_AUTO) {
      /* AUTO has no explicit LOD operand. Its group marker is emitted above. */
   } else if (d->lod == AGX3_TEX_LOD_BIAS) {
      if (d->array) {
         ov[n++] = (struct agx3_field_kv)KV(43, d->lod_x64);
         ov[n++] = (struct agx3_field_kv)KV(51, 1);
      } else {
         ov[n++] = (struct agx3_field_kv)KV(37, 1);
         ov[n++] = (struct agx3_field_kv)KV(29, d->lod_x64);
      }
      ov[n++] = (struct agx3_field_kv)KV(139, 1); /* bias mode */
   } else if (d->lod == AGX3_TEX_LOD_REGISTER) {
      ov[n++] = (struct agx3_field_kv)KV(28, 0); /* packed level bank */
      ov[n++] = (struct agx3_field_kv)KV(139, 4);
   } else if (d->lod == AGX3_TEX_LOD_BIAS_REGISTER) {
      ov[n++] = (struct agx3_field_kv)KV(28, 0); /* packed bias bank */
      ov[n++] = (struct agx3_field_kv)KV(139, 3);
   } else if (d->lod == AGX3_TEX_LOD_GRADIENT) {
      /* The gradient form clears the implicit-derivative marker: the four
       * staged scalars are the derivative. */
      ov[n++] = (struct agx3_field_kv)KV(139, 5);
   } else {
      if (d->array) {
         ov[n++] = (struct agx3_field_kv)KV(43, d->lod_x64);
         ov[n++] = (struct agx3_field_kv)KV(51, 1);
      } else {
         ov[n++] = (struct agx3_field_kv)KV(37, 1);
         ov[n++] = (struct agx3_field_kv)KV(29, d->lod_x64);
      }
      ov[n++] = (struct agx3_field_kv)KV(139, 2); /* explicit-level mode */
   }
   if (d->gather != AGX3_TEX_GATHER_NONE) {
      ov[n++] = (struct agx3_field_kv)KV(144, 1);
      if (d->gather != AGX3_TEX_GATHER_R)
         ov[n++] = (struct agx3_field_kv)KV(145, d->gather -
                                                 AGX3_TEX_GATHER_R);
   }
   if (d->shadow)
      ov[n++] = (struct agx3_field_kv)KV(143, 1);
   if (d->offset) {
      ov[n++] = (struct agx3_field_kv)KV(43, d->offset_xy);
      ov[n++] = (struct agx3_field_kv)KV(51, 1);
      ov[n++] = (struct agx3_field_kv)KV(141, 1);
   }
   if (enc == AGX_ISA_ENCODING_AGX3B)
      ov[n++] = (struct agx3_field_kv)KV(147, d->final ? 2 : 0);
   else
      ov[n++] = (struct agx3_field_kv)KV(150, d->final ? 2 : 0);
   for (unsigned i = 0; i < 8; ++i)
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(62 + i, 1);

   if (!agx3_bind_compose(enc, "smp", immediate ? 3 : 4, base, nb, ov, n, out, len)) {
      memset(out, 0, AGX3_TEX_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_tex_stage_copy_hints(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    uint8_t slot, uint16_t wait_mask,
                          bool src_cache, bool src_discard,
                    uint8_t out[AGX3_TEX_COPY_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TEX_COPY_MAX_LENGTH);
   *len = 0;
   if ((src_cache && src_discard) ||
       (enc == AGX_ISA_ENCODING_AGX3C && (src_cache || !src_discard)))
      return false;

   if (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C)
      return false;

   /* One even 32-bit register each way, as in every witness. The source field
    * is seven bits wide. Unlike PACK_SMPIR, COPY has no separately recovered
    * bit 7, so accepting a high-bank source would alias it into the low bank.
    */
   if ((dest & 1) || (src & 1) || dest >= 256 || src >= 128)
      return false;

   const unsigned slots = enc == AGX_ISA_ENCODING_AGX3C ? 8 : 6;
   const uint16_t waits = enc == AGX_ISA_ENCODING_AGX3C ? 0xff : 0x3f;
   if (slot < 1 || slot > slots || (wait_mask & ~waits) ||
       !one_or_zero(wait_mask))
      return false;

   struct agx3_field_kv base[8];
   unsigned nb = 0;
   base[nb++] = (struct agx3_field_kv)KV(9, 5);
   base[nb++] = (struct agx3_field_kv)KV(10, 2);
   base[nb++] = (struct agx3_field_kv)KV(23, 2);
   base[nb++] = (struct agx3_field_kv)KV(24, 2);
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(28, 1);
      base[nb++] = (struct agx3_field_kv)KV(29, 35);
   }

   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(5, dest);
   ov[n++] = (struct agx3_field_kv)KV(16, src);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);

   /* The compact form first; Apple escalates the encoding point when a slot
    * or wait bit does not fit, and the composer's first-success order
    * reproduces that choice on every witness. */
   for (int ep = 0; ep <= 1; ++ep) {
      if (agx3_bind_compose(enc, "copy", ep, base, nb, ov, n, out, len))
         return true;
   }
   memset(out, 0, AGX3_TEX_COPY_MAX_LENGTH);
   *len = 0;
   return false;
}

bool
agx3_tex_stage_copy(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    uint8_t slot, uint16_t wait_mask,
                    uint8_t out[AGX3_TEX_COPY_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_copy_hints(enc,dest,src,slot,wait_mask,false,true,out,len);
}

bool
agx3_tex_encode_image_write_rgba8_2d(
   enum agx_isa_encoding enc, const struct agx3_image_write_desc *d,
   uint8_t out[AGX3_IMAGE_WRITE_LENGTH], uint8_t *len)
{
   if (d == NULL || out == NULL || len == NULL)
      return false;

   memset(out, 0, AGX3_IMAGE_WRITE_LENGTH);
   *len = 0;
   /* This argument is the dependency consumed by the first data-staging COPY,
    * not an imgwci field. The fixed native instruction always waits for the
    * staging-slot2 writes, so both an outstanding slot1 producer and data
    * already retired by an earlier consumer use the same grounded bytes. */
   const unsigned kind =
      d->image_kind ? d->image_kind : AGX3_TEX_OPERAND_IMMEDIATE;
   const bool immediate = kind == AGX3_TEX_OPERAND_IMMEDIATE;
   const bool reg = kind == AGX3_TEX_OPERAND_REGISTER;
   if (enc != AGX_ISA_ENCODING_AGX3B ||
       (d->wait_mask & ~UINT16_C(0x3f)) != 0 ||
       (!immediate && !reg) ||
       (immediate && (d->image_kill ||
          !((d->image_value == 0 && !d->normalized) ||
            (d->image_value == 1 && d->normalized)))) ||
       /* f70 is the float-vs-integer bit and the state operand's form is a
        * different field family (f91 immediate, f92/f93 register); they are
        * orthogonal, so a register state does not imply normalized data.
        * metal-nt -arch applegpu_g17p on a dynamically indexed
        * `array<texture2d<T, access::write>, 4>`, one variable per kernel:
        *
        *   texture2d<float>  d72054000040001020040000883a0901
        *   texture2d<uint>   d72054000040001020040000881a0901
        *   texture2d<int>    d72054000040001020040000881a0901
        *
        * the state operand moves against the immediate-state controls and f70
        * does not, and int and uint are byte-identical. */
       (reg && ((d->image_value & 1) || d->image_value > 126)))
      return false;

   /* Nonzero field values recovered from native imgwci.ep0 instructions.
    * Every omitted field is intentionally zero. The direct state selector is
    * f91. Native argument-buffer indexing instead uses f92 for a killed GPR
    * or f93 for a retained GPR, with f99/f100 selecting that operand family.
    * The composer validates the complete selected form. */
   struct agx3_field_kv fields[24] = {
      KV(9, 5),   KV(12, 1),  KV(13, 1),   KV(24, 1), KV(38, 1),
      KV(51, 1),  KV(52, 1),  KV(60, 1),   KV(71, 2), KV(72, 1),
      KV(74, 1),  KV(85, 3),  KV(86, 3),   KV(113, 1),
      KV(70, d->normalized),
   };
   unsigned n = 15;
   if (immediate) {
      fields[n++] = (struct agx3_field_kv)KV(79, 4);
      fields[n++] = (struct agx3_field_kv)KV(91, d->image_value);
      fields[n++] = (struct agx3_field_kv)KV(99, 1);
      fields[n++] = (struct agx3_field_kv)KV(100, 1);
   } else if (d->image_kill) {
      fields[n++] = (struct agx3_field_kv)KV(92, d->image_value);
      fields[n++] = (struct agx3_field_kv)KV(99, 2);
      fields[n++] = (struct agx3_field_kv)KV(100, 2);
      fields[n++] = (struct agx3_field_kv)KV(103, 1);
   } else {
      fields[n++] = (struct agx3_field_kv)KV(93, d->image_value);
      fields[n++] = (struct agx3_field_kv)KV(99, 3);
      fields[n++] = (struct agx3_field_kv)KV(100, 2);
   }

   if (!agx3_bind_compose(enc, "imgwci", 0, fields,
                          n,
                          NULL, 0, out, len) ||
       *len != AGX3_IMAGE_WRITE_LENGTH) {
      memset(out, 0, AGX3_IMAGE_WRITE_LENGTH);
      *len = 0;
      return false;
   }

   return true;
}

static uint16_t
word(const uint8_t *p, unsigned i)
{
   return (uint16_t)p[2 * i] | ((uint16_t)p[2 * i + 1] << 8);
}

static bool
decode_b(const uint8_t *p, struct agx3_tex_fetch_desc *d)
{
   const uint16_t w0 = word(p, 0), w1 = word(p, 1), w2 = word(p, 2),
                  w4 = word(p, 4), w5 = word(p, 5);
   d->width = (w0 & 0x8000) ? AGX3_TEX_W32 : AGX3_TEX_W16;
   d->dest = d->width == AGX3_TEX_W32
                ? (((w0 >> 3) & 0x1f) << 1) | (((w1 >> 6) & 1) << 6)
                : ((w0 >> 3) & 0x1f) | (((w1 >> 6) & 3) << 5);
   d->coord = (((w0 >> 8) & 7) << 1) | (((w1 >> 8) & 3) << 4);
   d->coord_kill = (w1 >> 13) & 1;
   d->final = (w1 >> 15) & 1;

   const unsigned mask_code = ((w1 >> 11) & 3) | (((w2 >> 5) & 1) << 2) |
                              (((w5 >> 6) & 1) << 3);
   static const uint8_t masks[16] = {
      1, 2, 4, 8, 9, 3, 7, 15, 5, 6, 10, 12, 13, 11, 14, 0,
   };
   d->mask = masks[mask_code];
   if (!d->mask)
      return false;
   d->sampler_value = ((w4 >> 8) & 1) | ((w2 & 0xf) << 1) |
                      (((w4 >> 1) & 7) << 5);

   const unsigned slot_code = (w2 >> 10) & 7;
   if (slot_code > 5)
      return false;
   d->slot = slot_code + 1;
   d->texture = ((w4 >> 7) & 1) | (((w0 >> 11) & 0xf) << 1) |
                (((w4 >> 4) & 7) << 5);

   const unsigned wait_code = (w2 >> 7) & 7;
   if (wait_code > 6)
      return false;
   d->wait_mask = wait_code == 0 ? 0 : (1u << (wait_code - 1));
   return true;
}

static bool
decode_c(const uint8_t *p, struct agx3_tex_fetch_desc *d)
{
   const uint16_t w0 = word(p, 0), w1 = word(p, 1), w2 = word(p, 2),
                  w3 = word(p, 3);
   d->width = (w0 & 8) ? AGX3_TEX_W32 : AGX3_TEX_W16;
   d->dest = ((w3 >> 2) & 1) | (((w0 >> 4) & 0xf) << 1) |
             (((w1 >> 6) & 3) << 5) | (((w3 >> 12) & 1) << 7);
   d->coord = (((w1 >> 10) & 0x3f) << 2) | (((w1 >> 9) & 1) << 8);
   d->coord_kill = (w1 >> 4) & 1;
   d->final = (w2 >> 15) & 1;

   const unsigned mask_code = ((w3 >> 5) & 3) | (((w3 >> 11) & 1) << 2);
   if (mask_code == 0)
      d->mask = 1;
   else if (mask_code == 7)
      d->mask = 15;
   else
      return false;

   d->slot = ((w2 >> 3) & 3) | (((w3 >> 4) & 1) << 2);
   d->slot++;
   d->texture = ((w3 >> 3) & 1) | (((w2 >> 12) & 1) << 1) |
                (((w2 >> 13) & 3) << 2);
   const unsigned wait_code = (w3 >> 13) & 7;
   d->wait_mask = wait_code == 0 ? 0 : (1u << (wait_code - 1));
   return true;
}

bool
agx3_tex_decode_fetch(enum agx_isa_encoding enc, const void *instruction,
                      uint32_t instruction_size,
                      struct agx3_tex_fetch_desc *out)
{
   if (out == NULL)
      return false;
   memset(out, 0, sizeof(*out));
   if (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C)
      return false;
   if (enc == AGX_ISA_ENCODING_AGX3B && instruction && instruction_size == 20) {
      const uint8_t *p = instruction;
      struct agx3_tex_fetch_desc d = {0};
      if (!decode_b(p, &d))
         return false;
      d.texture = 0;
      d.texture_value = (((word(p, 0) >> 12) & 7) << 1) |
                        (((word(p, 4) >> 4) & 7) << 4) |
                        (((p[15] >> 2) & 1) << 7);
      d.wait_mask = ((p[4] >> 7) & 1) | ((p[5] & 3) << 1) |
                    (((p[18] >> 4) & 7) << 3);
      d.sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE;
      /* Repack both native index kinds. Their context selector, upper bits,
       * and every reserved field are validated by exact byte comparison. */
      for (unsigned kind = AGX3_TEX_OPERAND_REGISTER;
           kind <= AGX3_TEX_OPERAND_UNIFORM; ++kind) {
         d.texture_kind = kind;
         d.texture_kill = kind == AGX3_TEX_OPERAND_REGISTER && (p[15] & 8);
         for (unsigned array = 0; array < 2; ++array) {
            for (unsigned three_d = 0; three_d < 2; ++three_d) {
               d.array = array;
               d.three_d = three_d;
               uint8_t canonical[AGX3_TEX_MAX_LENGTH], length;
               if (agx3_tex_encode_fetch(enc, &d, canonical, &length) &&
                   length == instruction_size &&
                   !memcmp(canonical, instruction, length)) {
                  *out = d;
                  return true;
               }
            }
         }
      }
      return false;
   }
   const uint32_t expected = enc == AGX_ISA_ENCODING_AGX3B ? 14 : 8;
   if (instruction == NULL || instruction_size != expected)
      return false;

   struct agx3_tex_fetch_desc d = {0};
   const uint8_t *p = instruction;
   if (!(enc == AGX_ISA_ENCODING_AGX3B ? decode_b(p, &d) : decode_c(p, &d)) ||
       !fetch_ok(enc, &d))
      return false;

   /* The array and 3D axes are selectors the hand decoders above do not
    * extract. Recover them the same way everything else here is validated,
    * by requiring one reading to repack byte-for-byte. */
   for (unsigned a = 0; a < 2; ++a) {
      for (unsigned three_d = 0; three_d < 2; ++three_d) {
         uint8_t canonical[AGX3_TEX_MAX_LENGTH];
         uint8_t length = 0;
         d.array = a != 0;
         d.three_d = three_d != 0;
         if (agx3_tex_encode_fetch(enc, &d, canonical, &length) &&
             length == instruction_size &&
             memcmp(canonical, instruction, length) == 0) {
            *out = d;
            return true;
         }
      }
   }

   return false;
}

/* Fresh native G17P LOD fragment witnesses use this 12-byte PACK_SMPIR for a
 * scalar LOD, preserving the source when it remains live. f5=6 selects
 * operand bank 6, not architectural r3. The FLOAT source is a half, while the
 * INTEGER source is the full uint required by texture2d::read. */
bool
agx3_tex_stage_typed_lod(enum agx_isa_encoding enc, uint16_t src, uint8_t slot,
                         uint16_t wait_mask, bool src_cache, bool src_discard,
                         enum agx3_tex_level_kind kind,
                         uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || src >= 256 || slot < 1 || slot > 6 ||
       (wait_mask & ~0x3f) || (src_cache && src_discard) ||
       (kind != AGX3_TEX_LEVEL_FLOAT && kind != AGX3_TEX_LEVEL_INTEGER))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 6), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(74, (unsigned)kind), KV(75, 16),
   };
   struct agx3_field_kv ov[10];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, src & 0x7f);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   if (!agx3_bind_compose(enc, "pack_smpir", 0, base,
                         sizeof(base) / sizeof(base[0]), ov, n, out, len))
      return false;

   /* The source is an eight-bit half-register index (f16 bits 0..6 at
    * instruction bits 41..47, bit 7 at bit 48). Apple's encoder's 191
    * bound is a compiler allocation policy, not a narrower wire field.
    * Keep the same native instruction for Mesa's full 256-half bank. */
   out[6] |= (src >> 7) & 1;
   return true;
}

bool
agx3_tex_stage_lod(enum agx_isa_encoding enc, uint16_t src, uint8_t slot,
                   uint16_t wait_mask, bool src_cache, bool src_discard,
                   uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_typed_lod(enc, src, slot, wait_mask, src_cache,
                                   src_discard, AGX3_TEX_LEVEL_FLOAT, out,
                                   len);
}

/* The dynamic array-layer witness uses PACK_SMPIR like register LOD, but its
 * operand-type selectors differ. In particular, f74=4/f75=0 produces the
 * native 1705540600080000f0c00400 for discarded r2 on slot 1; the LOD
 * baseline's f74=2/f75=16 silently leaves the sample destination unwritten.
 * Register-indexed array samples use the independently witnessed slot-6 form
 * 1705540600180000f0e40400 for discarded r6. */
bool
agx3_tex_stage_array_layer(enum agx_isa_encoding enc, uint16_t src,
                           uint8_t slot, uint16_t wait_mask, bool src_cache,
                           bool src_discard,
                           uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || src >= 256 || slot < 1 || slot > 6 ||
       (wait_mask & ~0x3f) || (src_cache && src_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 6), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(74, 4),
   };
   struct agx3_field_kv ov[10];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, src & 0x7f);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   if (!agx3_bind_compose(enc, "pack_smpir", 0, base,
                         sizeof(base) / sizeof(base[0]), ov, n, out, len))
      return false;

   out[6] |= (src >> 7) & 1;
   return true;
}

/* Apple's G17P cube path first runs one CEM calculation for each axis.  The
 * three source fields are the input XYZ half-register indices; axis 2 is the
 * last consumer in the witnessed sequence and carries their discard bits.
 *
 * f_plain from the native cube corpus, with x/y/z in 4/6/20, is exactly:
 *
 *   axis 0  1701540003040d5254219200
 *   axis 1  1701540703040d5254219204
 *   axis 2  1701540203040c50502f9208
 */
bool
agx3_tex_cube_cem_calc_to(enum agx_isa_encoding enc, uint16_t x, uint16_t y,
                          uint16_t z, uint16_t dest, uint8_t axis,
                          bool discard,
                          uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || x >= 128 || y >= 128 || z >= 128 ||
       (dest & 1) || dest > 190 || axis > 2 || (discard && axis != 2))
      return false;

   const struct agx3_field_kv base[] = {
      KV(9, 2),  KV(10, 2), KV(11, 1), KV(23, 2), KV(24, 2),
      KV(40, 2), KV(41, 2), KV(57, 2), KV(58, 2),
   };
   struct agx3_field_kv ov[10];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, x);
   ov[n++] = (struct agx3_field_kv)KV(33, y);
   ov[n++] = (struct agx3_field_kv)KV(50, z);
   ov[n++] = (struct agx3_field_kv)KV(75, axis);
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   if (discard) {
      ov[n++] = (struct agx3_field_kv)KV(27, 1);
      ov[n++] = (struct agx3_field_kv)KV(44, 1);
      ov[n++] = (struct agx3_field_kv)KV(61, 1);
   } else if (axis != 2) {
      ov[n++] = (struct agx3_field_kv)KV(25, 1);
      ov[n++] = (struct agx3_field_kv)KV(42, 1);
      ov[n++] = (struct agx3_field_kv)KV(59, 1);
   }

   return agx3_bind_compose(enc, "cem_calc", 0, base,
                            sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

bool
agx3_tex_cube_cem_calc(enum agx_isa_encoding enc, uint16_t x, uint16_t y,
                       uint16_t z, uint8_t axis, bool discard,
                       uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   const uint16_t dest[3] = {0, 14, 4};
   if (axis > 2)
      return false;
   return agx3_tex_cube_cem_calc_to(enc, x, y, z, dest[axis], axis, discard,
                                    out, len);
}

/* CEM axis 0 leaves the major-axis magnitude in its destination. Apple's cube
 * shaders apply an in-place reciprocal before calculating the two normalized
 * face coordinates. */
bool
agx3_tex_cube_cem_rcp_to(enum agx_isa_encoding enc, uint16_t value,
                         uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   return agx3_alu_f32rcp(enc, value, value, AGX3_ALU_HINT_DISCARD, true, 0,
                          out, len);
}

bool
agx3_tex_cube_cem_rcp(enum agx_isa_encoding enc,
                      uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_cube_cem_rcp_to(enc, 0, out, len);
}

/* CEM leaves the signed face coordinates in its axis destinations. Apple's
 * native cube shaders finish them with this special FMA form: f35 names the
 * reciprocal major axis, f18 names the selected axis value and f12 groups the
 * result with the face PACK_SMPIR transaction. */
bool
agx3_tex_cube_cem_finalize_to(enum agx_isa_encoding enc, uint8_t component,
                              uint16_t reciprocal, uint16_t axis_value,
                              uint8_t slot,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || component > 1 ||
       (reciprocal & 1) || reciprocal > 190 || (axis_value & 1) ||
       axis_value > 190 || slot < 1 || slot > 6)
      return false;

   const struct agx3_field_kv base0[] = {
      KV(9, 5),   KV(10, 2), KV(14, 1), KV(25, 2), KV(26, 2),
      KV(29, 1),  KV(42, 2), KV(43, 2), KV(44, 1), KV(47, 1),
      KV(51, 32), KV(59, 1), KV(60, 2), KV(77, 2), KV(91, 2),
   };
   const struct agx3_field_kv base1[] = {
      KV(5, 2),  KV(9, 5),   KV(10, 2), KV(14, 1), KV(25, 2),
      KV(26, 2), KV(29, 1),  KV(42, 2), KV(43, 2), KV(46, 1),
      KV(47, 1), KV(51, 32), KV(59, 1), KV(60, 2), KV(77, 2),
      KV(91, 2),
   };
   const struct agx3_field_kv ov[] = {
      KV(12, slot), KV(18, axis_value), KV(35, reciprocal),
   };
   return agx3_bind_compose(
      enc, "f32fma", 5, component ? base1 : base0,
      component ? sizeof(base1) / sizeof(base1[0])
                : sizeof(base0) / sizeof(base0[0]),
      ov, sizeof(ov) / sizeof(ov[0]), out, len);
}

bool
agx3_tex_cube_cem_finalize(enum agx_isa_encoding enc, uint8_t component,
                           bool discard,
                           uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (component > 1 || (component == 0 && discard))
      return false;
   const uint16_t axis_value[2] = {14, 4};
   if (!agx3_tex_cube_cem_finalize_to(enc, component, 0,
                                      axis_value[component], 1, out, len))
      return false;
   if (discard)
      out[10] |= 0x60;
   return true;
}

/* CEM publishes the selected face through the FLOAT cube form of PACK_SMPIR.
 * For discarded special r1 on slot 1 Apple emits exactly
 * 1705540600040004f0c00000; only f74 differs from the array-layer form. */
bool
agx3_tex_stage_cube_z(enum agx_isa_encoding enc, uint16_t src,
                      uint8_t slot, uint16_t wait_mask, bool src_cache,
                      bool src_discard,
                      uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || src >= 256 || (src & 1) ||
       slot < 1 || slot > 6 || (wait_mask & ~0x3f) ||
       !one_or_zero(wait_mask) || (src_cache && src_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 6), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(74, 10),
   };
   struct agx3_field_kv ov[10];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, src & 0x7f);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   if (!agx3_bind_compose(enc, "pack_smpir", 0, base,
                         sizeof(base) / sizeof(base[0]), ov, n, out, len))
      return false;

   out[6] |= (src >> 7) & 1;
   return true;
}

/* depthcube_array::sample_compare combines CEM's selected face and the
 * integer array layer in one PACK_SMPIR form. applegpu-nt 32023.917 emits
 * exactly 1705540600060800f0c02900 for discarded 32-bit r3 on slot 1.
 *
 * Unlike the ordinary scalar PACK forms, f16 names a full 32-bit register,
 * not a half-register. The five additional selectors below are the exact
 * delta which selects the combined face/layer operand in Apple's table.
 */
bool
agx3_tex_stage_cube_array_shadow(enum agx_isa_encoding enc, uint16_t layer,
                                 uint8_t slot, uint16_t wait_mask,
                                 bool layer_cache, bool layer_discard,
                                 uint8_t out[AGX3_TEX_MAX_LENGTH],
                                 uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || layer >= 256 || (layer & 1) ||
       slot < 1 || slot > 6 || (wait_mask & ~0x3f) ||
       !one_or_zero(wait_mask) || (layer_cache && layer_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 6),  KV(9, 5),  KV(10, 2), KV(23, 2), KV(24, 1),
      KV(30, 2), KV(37, 2), KV(38, 1), KV(41, 1), KV(51, 1),
      KV(52, 2), KV(74, 4), KV(76, 10), KV(77, 12),
   };
   struct agx3_field_kv ov[10];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, layer >> 1);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, layer_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, layer_discard);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);

   return agx3_bind_compose(enc, "pack_smpir", 0, base,
                            sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

/* Own-MSL texture2d_ms reads publish a variable integer sample with this
 * distinct PACK_SMPIR before the otherwise sample-0 MS fetch. The exact
 * discarded-r2h witness is 1705540400000000d0c00800. */
bool
agx3_tex_stage_ms_sample(enum agx_isa_encoding enc, uint16_t src,
                         uint16_t wait_mask, bool src_cache,
                         bool src_discard,
                         uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || src >= 256 ||
       (wait_mask & ~0x3f) || (src_cache && src_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(9, 5),  KV(10, 2), KV(12, 1), KV(23, 2), KV(24, 1), KV(37, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(74, 9), KV(75, 12),
      KV(76, 1), KV(77, 16),
   };
   struct agx3_field_kv ov[10];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(5, src);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);

   return agx3_bind_compose(enc, "pack_smpir", 0, base,
                            sizeof(base) / sizeof(base[0]), ov, n, out, len);
}
