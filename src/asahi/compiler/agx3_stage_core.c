/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_stage_core.h"

#include <string.h>

#include "agx3_binder.h"
#include "agx3_tex_core.h"

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

bool
agx3_stage_vertex_alu(enum agx_isa_encoding enc, bool fma,
                      uint16_t dest, uint8_t slot, const uint16_t src[3],
                      bool uniform0, const bool live[3], const bool discard[3],
                      uint16_t wait_mask, uint8_t out[16], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !live || !discard ||
       !out || !len || dest >= 256 || (dest & 1) || slot < 1 || slot > 6 ||
       wait_mask > 0x3f || (uniform0 && !fma))
      return false;
   for (unsigned s = 0; s < (fma ? 3 : 2); ++s) {
      if (src[s] >= 256 || (src[s] & 1) || (live[s] && discard[s]))
         return false;
   }
   if (uniform0 && (live[0] || discard[0]))
      return false;

   /* Exact native witnesses are recorded with their full field vectors in
    * vertex/native-terrain-fields.json. 0x360 is a staged FMUL, 0x350 a
    * uniform/register/register FMA, and 0x404 an all-register FMA. */
   const struct agx3_field_kv base[] = {
      KV(9, 5), KV(10, 2), KV(25, 2), KV(26, 2), KV(43, 2),
      KV(60, 2),
   };
   struct agx3_field_kv fields[24] = {
      KV(5, dest), KV(12, slot), KV(18, src[1]),
      KV(42, uniform0 ? 3 : 2), KV(uniform0 ? 36 : 35, src[0]),
      KV(27, live[1]), KV(29, discard[1]),
   };
   unsigned n = 7;
   if (!uniform0) {
      fields[n++] = (struct agx3_field_kv)KV(44, live[0]);
      fields[n++] = (struct agx3_field_kv)KV(46, discard[0]);
   }
   if (fma) {
      fields[n++] = (struct agx3_field_kv)KV(52, src[2]);
      fields[n++] = (struct agx3_field_kv)KV(59, 2);
      fields[n++] = (struct agx3_field_kv)KV(61, live[2]);
      fields[n++] = (struct agx3_field_kv)KV(63, discard[2]);
      fields[n++] = (struct agx3_field_kv)KV(77, 2);
      fields[n++] = (struct agx3_field_kv)KV(91, 2);
   } else {
      fields[n++] = (struct agx3_field_kv)KV(74, 2);
   }
   for (unsigned s = 0; s < 6; ++s)
      fields[n++] = (struct agx3_field_kv)KV((fma ? 97 : 80) + s,
                                           (wait_mask >> s) & 1);
   for (unsigned ep = 0; ep <= (fma ? 5 : 3); ++ep) {
      if (agx3_bind_compose(enc, fma ? "f32fma" : "f32mul", ep,
                           base, 6, fields, n, out, len))
         return true;
   }
   return false;
}

static bool
b_only(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3B;
}

/* The staging destination class: index in f5, kill in f13, slot in f12. */
static unsigned
append_stage_dest(struct agx3_field_kv *ov, unsigned n, uint16_t dest,
                  bool kill, uint8_t slot)
{
   ov[n++] = (struct agx3_field_kv)KV(9, 5);
   ov[n++] = (struct agx3_field_kv)KV(5, dest);
   ov[n++] = (struct agx3_field_kv)KV(13, kill ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   return n;
}

/*
 * staging[dest] <- reg[src], 32-bit, taking scoreboard slot `slot`.
 *
 * This is exactly agx3_tex_stage_copy; it is re-exported here so the writeout
 * path does not have to include the texture header to reach a general encoder.
 */
static bool
agx3_stage_copy(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                uint8_t slot, uint16_t wait_mask,
                uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len)
{
   /* Same encoder, same witnesses; only the name is neutral. */
   return agx3_tex_stage_copy(enc, dest, src, slot, wait_mask, out, len);
}

bool
agx3_stage_copy_source(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                        bool src_kill, uint8_t slot, uint16_t wait_mask,
                        uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len)
{
   return agx3_stage_copy_liveness(enc, dest, src, false, src_kill, slot,
                                   wait_mask, out, len);
}

bool
agx3_stage_copy_liveness(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                          bool stage_kill, bool src_kill, uint8_t slot,
                          uint16_t wait_mask,
                          uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len)
{
   const unsigned slots = enc == AGX_ISA_ENCODING_AGX3C ? 8 : 6;
   if (!out || !len ||
       (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
       (stage_kill && enc != AGX_ISA_ENCODING_AGX3B) ||
       (dest & 1) || (src & 1) || dest >= 256 || src >= 256 ||
       slot < 1 || slot > slots || (wait_mask & ~((1u << slots) - 1)))
      return false;

   struct agx3_field_kv base[7] = {
      KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 2), KV(27, src_kill),
   };
   unsigned nb = 5;
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(28, 1);
      base[nb++] = (struct agx3_field_kv)KV(29, 35);
   }
   struct agx3_field_kv ov[12] = {
      KV(5, dest), KV(16, src), KV(12, slot), KV(13, stage_kill),
   };
   unsigned n = 4;
   for (unsigned i = 0; i < slots; ++i) {
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);
   }

   for (int ep = 0; ep <= 1; ++ep) {
      if (agx3_bind_compose(enc, "copy", ep, base, nb, ov, n, out, len))
         return true;
   }
   return false;
}

bool
agx3_stage_copy_kill(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     bool kill, uint8_t slot, uint16_t wait_mask,
                     uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len)
{
   if (!kill)
      return agx3_stage_copy(enc, dest, src, slot, wait_mask, out, len);

   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_STAGE_MAX_LENGTH);
   *len = 0;

   if (!b_only(enc) || (dest & 1) || (src & 1) || dest >= 256 || src >= 256)
      return false;
   if (slot < 1 || slot > 6 || (wait_mask & ~(uint16_t)0x3f))
      return false;

   /* f27 = 1 accompanies the 32-bit register source, exactly as in the
    * texture-coordinate copy this shares an encoding with. */
   const struct agx3_field_kv base[] = {KV(10, 2), KV(23, 2), KV(24, 2),
                                        KV(27, 1)};

   struct agx3_field_kv ov[14];
   unsigned n = append_stage_dest(ov, 0, dest, true, slot);
   ov[n++] = (struct agx3_field_kv)KV(16, src);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);

   /* The kill bit does not fit the compact form, so Apple escalates to ep1;
    * the composer's first-success order reproduces that. */
   if (agx3_bind_compose(enc, "copy", 0, base, 4, ov, n, out, len))
      return true;
   if (agx3_bind_compose(enc, "copy", 1, base, 4, ov, n, out, len))
      return true;

   memset(out, 0, AGX3_STAGE_MAX_LENGTH);
   *len = 0;
   return false;
}

static bool
clr_common(enum agx_isa_encoding enc, const struct agx3_field_kv *ov,
           unsigned n, uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len)
{
   const struct agx3_field_kv base[] = {KV(10, 2)};
   if (agx3_bind_compose(enc, "clr", 0, base, 1, ov, n, out, len))
      return true;
   memset(out, 0, AGX3_STAGE_MAX_LENGTH);
   *len = 0;
   return false;
}

bool
agx3_clr(enum agx_isa_encoding enc, uint16_t dest, bool half, bool hint,
         uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_STAGE_MAX_LENGTH);
   *len = 0;

   if (!b_only(enc) || dest >= 256 || (!half && (dest & 1)))
      return false;

   struct agx3_field_kv ov[5];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(9, 2);
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(11, hint ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(10, half ? 1 : 2);
   return clr_common(enc, ov, n, out, len);
}
