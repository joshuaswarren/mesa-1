/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_tile_core.h"

#include "agx3_binder.h"

#include <string.h>

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

static bool
is_b(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3B;
}

static bool
one_or_zero(uint16_t value)
{
   return (value & (value - 1)) == 0;
}

/*
 * The sample-mask operand. The immediate alternative is witnessed at every 1x
 * width; the register alternative is witnessed at 2x and 4x on applegpu_g17p
 * (AGX3b). On AGX3c/AGX3d the register form is admitted only when the caller
 * opts in (g17_register): the G18G/G17G 4x per-pixel store witness is
 * 270605082901ad02000071440000 = {f44:10, f51:2, f55:1, wait f18}
 * (native-msaa-reference-147), and nothing else is measured there.
 */
static bool
sample_ok(enum agx_isa_encoding enc, struct agx3_tile_sample s)
{
   if (s.reg)
      return (is_b(enc) ||
              (s.g17_register && agx_encoding_is_g17_class(enc))) &&
             s.value < 192;

   return !s.kill && s.value <= 255;
}

static unsigned
sample_overrides(struct agx3_tile_sample s, struct agx3_field_kv *ov,
                 unsigned n)
{
   if (s.reg) {
      ov[n++] = (struct agx3_field_kv)KV(51, 2);
      ov[n++] = (struct agx3_field_kv)KV(44, s.value);
      if (s.kill)
         ov[n++] = (struct agx3_field_kv)KV(55, 1);
   } else {
      ov[n++] = (struct agx3_field_kv)KV(51, 1);
      ov[n++] = (struct agx3_field_kv)KV(43, s.value);
   }

   return n;
}

static bool
common_ok(uint16_t reg, enum agx3_tile_width w, uint8_t mask,
          struct agx3_tile_sample sample, enum agx_isa_encoding enc,
          uint16_t wait_mask, uint16_t tile_off)
{
   const unsigned nwait = is_b(enc) ? 6 : 8;
   return (enc == AGX_ISA_ENCODING_AGX3B ||
           agx_encoding_is_g17_class(enc)) &&
          reg < (is_b(enc) ? 192 : 256) &&
          (w != AGX3_TILE_W32 || (reg & 1) == 0) &&
          (w == AGX3_TILE_W8 || w == AGX3_TILE_W16 || w == AGX3_TILE_W32) &&
          mask >= 1 && mask <= 15 && sample_ok(enc, sample) &&
          tile_off <= UINT8_MAX &&
          (wait_mask & ~((1u << nwait) - 1)) == 0 &&
          one_or_zero(wait_mask);
}

static unsigned
wait_overrides(enum agx_isa_encoding enc, uint16_t wait_mask,
               struct agx3_field_kv *ov, unsigned n)
{
   /* Wait bits are f18..f25 (slots 1..8). */
   for (unsigned i = 0; i < (is_b(enc) ? 6 : 8); ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(18 + i, 1);
   return n;
}

bool
agx3_tile_store(enum agx_isa_encoding enc, uint16_t data,
                enum agx3_tile_width width, uint8_t chan_mask,
                uint16_t tile_off, struct agx3_tile_sample sample,
                uint16_t wait_mask, uint8_t out[AGX3_TILE_MAX_LENGTH],
                uint8_t *len)
{
   return agx3_tile_store_f13(enc, data, width, chan_mask, tile_off, sample,
                              wait_mask, false, out, len);
}

bool
agx3_tile_store_f13(enum agx_isa_encoding enc, uint16_t data,
                    enum agx3_tile_width width, uint8_t chan_mask,
                    uint16_t tile_off, struct agx3_tile_sample sample,
                    uint16_t wait_mask, bool f13,
                    uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TILE_MAX_LENGTH);
   *len = 0;

   if (!common_ok(data, width, chan_mask, sample, enc, wait_mask, tile_off))
      return false;
   /* G18G (agx3d) keeps lstcol bit for bit: the 96 aligned
    * applegpu_g17g/applegpu_g18g lstcol words of the m5 reference corpus are
    * byte-identical, as is the oracle colour store 27 01 05 08 21 01 ad 02 00
    * 00 61 5c 00 00; the wait field stays a mask. */
   const struct agx3_field_kv base_b[] = {KV(9, 2),  KV(35, 9), KV(37, 7),
                                          KV(38, 1), KV(52, 1), KV(65, 1),
                                          KV(66, 1)};
   const struct agx3_field_kv base_c[] = {
      KV(9, 2),  KV(17, 111), KV(35, 9), KV(37, 7), KV(38, 1),
      KV(52, 1), KV(65, 1),   KV(66, 1)};
   struct agx3_field_kv ov[16];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, data);
   /* The explicit C/D register-sample path receives the caller's data
    * lifetime bit. Preserve legacy whole-pixel and G17P behavior. */
   if (f13 || (!is_b(enc) && !sample.g17_register))
      ov[n++] = (struct agx3_field_kv)KV(13, 1);
   ov[n++] = (struct agx3_field_kv)KV(10, width);
   ov[n++] = (struct agx3_field_kv)KV(26, chan_mask);
   ov[n++] = (struct agx3_field_kv)KV(57, tile_off);
   n = sample_overrides(sample, ov, n);
   n = wait_overrides(enc, wait_mask, ov, n);
   if (!agx3_bind_compose(enc, "lstcol", agx3_bind_ep(enc, 0),
                          is_b(enc) ? base_b : base_c, is_b(enc) ? 7 : 8, ov,
                          n, out, len)) {
      memset(out, 0, AGX3_TILE_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_tile_load(enum agx_isa_encoding enc, uint16_t dest,
               enum agx3_tile_width width, uint8_t slot, uint8_t chan_mask,
               uint16_t tile_off, struct agx3_tile_sample sample,
               uint16_t wait_mask, bool first, bool last,
               uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TILE_MAX_LENGTH);
   *len = 0;

   /* lldcol gains f79 on G18G (the one aligned applegpu_g18g tile load has
    * f79 = 2, its G17G twin none), which is not modelled: refuse agx3d. */
   if (enc == AGX_ISA_ENCODING_AGX3D)
      return false;

   /* Native454/513 ground raw32 reloads; native567 additionally grounds
    * packed BGRA8 and RGBA16Float register-sample reloads. Their full
    * tuples occupy four, eight or sixteen bytes according to the width.
    * The G18 long-form field is supplied by the transcode rule. */
   bool scalar_ms = sample.g17_register && sample.reg &&
                    enc == AGX_ISA_ENCODING_AGX3C &&
                    width == AGX3_TILE_W32 && chan_mask == 1 &&
                    !(tile_off % 4);
   if (sample.g17_register &&
       (enc != AGX_ISA_ENCODING_AGX3C || !sample.reg ||
        (width != AGX3_TILE_W8 && width != AGX3_TILE_W16 &&
         width != AGX3_TILE_W32) ||
        (!scalar_ms &&
         (chan_mask != 15 || (tile_off % (4u << width))))))
      return false;

   /* Full-component loads, plus the G17P scalar, pair, and RGB565 forms.
    * Fresh applegpu_g17p RG8, RG16{Float,Unorm,Snorm,Uint,Sint}, and
    * RG32{Float,Uint,Sint} render-pipeline witnesses isolate mask3 in f26 at
    * every width. A B5G6R5Unorm framebuffer-fetch witness independently
    * isolates mask7 in f26 for W16:
    *
    *    670e54000000019a02000000
    *
    * None carries an incoming wait. */
   if (!common_ok(dest, width, chan_mask, sample, enc, wait_mask, tile_off) ||
       (chan_mask != 15 && !scalar_ms &&
        !(is_b(enc) &&
          (chan_mask == 1 || chan_mask == 3 ||
           (chan_mask == 7 && width == AGX3_TILE_W16)))) ||
       wait_mask != 0 || slot < 1 ||
       slot > (is_b(enc) ? 6 : 8))
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 2),  KV(35, 9), KV(37, 7),
                                          KV(38, 1), KV(52, 1), KV(65, 1),
                                          KV(66, 1)};
   const struct agx3_field_kv base_c[] = {KV(9, 2),  KV(17, 110), KV(35, 9),
                                          KV(37, 7), KV(38, 1),   KV(52, 1),
                                          KV(65, 1), KV(66, 1)};
   struct agx3_field_kv ov[16];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(10, width);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(26, chan_mask);
   ov[n++] = (struct agx3_field_kv)KV(57, tile_off);
   n = sample_overrides(sample, ov, n);
   ov[n++] = (struct agx3_field_kv)KV(78, first ? 1 : 0);
   if (!is_b(enc))
      ov[n++] = (struct agx3_field_kv)KV(74, last ? 2 : 0);
   if (!agx3_bind_compose(enc, "lldcol", 0, is_b(enc) ? base_b : base_c,
                          is_b(enc) ? 7 : 8, ov, n, out, len)) {
      memset(out, 0, AGX3_TILE_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}
