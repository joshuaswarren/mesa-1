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

static bool
coverage_ok(enum agx_isa_encoding enc, enum agx3_tile_coverage cov)
{
   return cov == AGX3_TILE_COV_1X ||
          (is_b(enc) && cov >= AGX3_TILE_COV_2X_BROADCAST &&
           cov <= AGX3_TILE_COV_4X_SAMPLE_3);
}

static bool
common_ok(uint16_t reg, enum agx3_tile_width w, uint8_t mask,
          enum agx3_tile_coverage cov, enum agx_isa_encoding enc,
          uint16_t wait_mask, uint16_t tile_off)
{
   const unsigned nwait = is_b(enc) ? 6 : 8;
   return (enc == AGX_ISA_ENCODING_AGX3B ||
           enc == AGX_ISA_ENCODING_AGX3C) &&
          reg < (is_b(enc) ? 192 : 256) &&
          (w != AGX3_TILE_W32 || (reg & 1) == 0) &&
          (w == AGX3_TILE_W8 || w == AGX3_TILE_W16 || w == AGX3_TILE_W32) &&
          mask >= 1 && mask <= 15 && coverage_ok(enc, cov) &&
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
                uint16_t tile_off, enum agx3_tile_coverage cov,
                uint16_t wait_mask, uint8_t out[AGX3_TILE_MAX_LENGTH],
                uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TILE_MAX_LENGTH);
   *len = 0;

   if (!common_ok(data, width, chan_mask, cov, enc, wait_mask, tile_off))
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 2),  KV(35, 9), KV(37, 7),
                                          KV(38, 1), KV(52, 1), KV(65, 1),
                                          KV(66, 1)};
   const struct agx3_field_kv base_c[] = {
      KV(9, 2),  KV(17, 111), KV(35, 9), KV(37, 7), KV(38, 1),
      KV(52, 1), KV(65, 1),   KV(66, 1)};
   struct agx3_field_kv ov[16];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, data);
   if (!is_b(enc))
      ov[n++] = (struct agx3_field_kv)KV(13, 1);
   ov[n++] = (struct agx3_field_kv)KV(10, width);
   ov[n++] = (struct agx3_field_kv)KV(26, chan_mask);
   ov[n++] = (struct agx3_field_kv)KV(57, tile_off);
   unsigned group = 1, selector = 1;
   switch (cov) {
   case AGX3_TILE_COV_1X:
      break;
   case AGX3_TILE_COV_2X_BROADCAST:
   case AGX3_TILE_COV_4X_BROADCAST:
      group = 2;
      selector = 10;
      break;
   case AGX3_TILE_COV_2X_SAMPLE_0:
   case AGX3_TILE_COV_2X_SAMPLE_1:
      group = 2;
      selector = 4 + (cov - AGX3_TILE_COV_2X_SAMPLE_0);
      break;
   case AGX3_TILE_COV_4X_SAMPLE_0:
   case AGX3_TILE_COV_4X_SAMPLE_1:
   case AGX3_TILE_COV_4X_SAMPLE_2:
   case AGX3_TILE_COV_4X_SAMPLE_3:
      group = 2;
      selector = 8 + (cov - AGX3_TILE_COV_4X_SAMPLE_0);
      break;
   default:
      return false;
   }
   ov[n++] = (struct agx3_field_kv)KV(51, group);
   ov[n++] = (struct agx3_field_kv)KV(group == 1 ? 43 : 44, selector);
   if (group == 2)
      ov[n++] = (struct agx3_field_kv)KV(55, 1);
   n = wait_overrides(enc, wait_mask, ov, n);
   if (!agx3_bind_compose(enc, "lstcol", 0, is_b(enc) ? base_b : base_c,
                          is_b(enc) ? 7 : 8, ov, n, out, len)) {
      memset(out, 0, AGX3_TILE_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_tile_load(enum agx_isa_encoding enc, uint16_t dest,
               enum agx3_tile_width width, uint8_t slot, uint8_t chan_mask,
               uint16_t tile_off, enum agx3_tile_coverage cov,
               uint16_t wait_mask, bool first, bool last,
               uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TILE_MAX_LENGTH);
   *len = 0;

   /* Full-component loads, plus the G17P scalar and pair forms. Fresh
    * applegpu_g17p RG8, RG16{Float,Unorm,Snorm,Uint,Sint}, and
    * RG32{Float,Uint,Sint} render-pipeline witnesses isolate mask3 in f26 at
    * every width. None carries an incoming wait. */
   if (!common_ok(dest, width, chan_mask, cov, enc, wait_mask, tile_off) ||
       (chan_mask != 15 &&
        !(is_b(enc) && (chan_mask == 1 || chan_mask == 3))) ||
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
   unsigned selector = 1;
   switch (cov) {
   case AGX3_TILE_COV_1X:
      break;
   case AGX3_TILE_COV_2X_SAMPLE_0:
   case AGX3_TILE_COV_2X_SAMPLE_1:
      selector = 1u << (cov - AGX3_TILE_COV_2X_SAMPLE_0);
      break;
   case AGX3_TILE_COV_4X_SAMPLE_0:
   case AGX3_TILE_COV_4X_SAMPLE_1:
   case AGX3_TILE_COV_4X_SAMPLE_2:
   case AGX3_TILE_COV_4X_SAMPLE_3:
      selector = 1u << (cov - AGX3_TILE_COV_4X_SAMPLE_0);
      break;
   case AGX3_TILE_COV_2X_BROADCAST:
   case AGX3_TILE_COV_4X_BROADCAST:
   default:
      /* Native lldcol has no witnessed all-sample reduction form. */
      return false;
   }
   ov[n++] = (struct agx3_field_kv)KV(51, 1);
   ov[n++] = (struct agx3_field_kv)KV(43, selector);
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
