/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_vary_core.h"

#include "agx3_binder.h"

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

static bool
is_b(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3B;
}

static bool
agx3_vary_interp_mode(enum agx_isa_encoding enc, uint16_t dest,
                      uint16_t coef_index, bool perspective,
                      enum agx3_vary_location location, bool first,
                      bool projective_rhw, bool explicit_sample,
                      uint16_t coords, bool coords_kill,
                      uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   if ((enc != AGX_ISA_ENCODING_AGX3B &&
        enc != AGX_ISA_ENCODING_AGX3C) ||
       (first && enc != AGX_ISA_ENCODING_AGX3B) ||
       (projective_rhw && (enc != AGX_ISA_ENCODING_AGX3B ||
                           (!explicit_sample && location != AGX3_VARY_CENTER) ||
                           !perspective || coef_index > 0x7f)) ||
       (explicit_sample && (enc != AGX_ISA_ENCODING_AGX3B || first ||
                            (coords & 1) != 0 || coords >= 256)) ||
       (dest & 1) != 0 || dest >= 256 ||
       (location != AGX3_VARY_CENTER && location != AGX3_VARY_CENTROID))
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 2), KV(10, 2), KV(25, 6),
                                          KV(26, 2), KV(40, 2)};
   const struct agx3_field_kv base_c[] = {KV(9, 2),  KV(10, 2), KV(25, 6),
                                          KV(26, 2), KV(40, 2), KV(44, 3),
                                          KV(45, 58)};
   struct agx3_field_kv ov[9];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(22, coef_index);
   ov[n++] = (struct agx3_field_kv)KV(11, perspective ? 1 : 0);
   if (explicit_sample) {
      /* Apple Metal 32023.917, applegpu_g17p. The source SHA-256
       * 2f0835ef... produces perspective, no-perspective, and mixed
       * center/sample programs:
       * f39/f40 select a 32-bit coordinate register, f32 names the token,
       * and f41/f43 are its live/final-use alternatives. */
      ov[n++] = (struct agx3_field_kv)KV(39, 2);
      ov[n++] = (struct agx3_field_kv)KV(32, coords);
      ov[n++] = (struct agx3_field_kv)KV(coords_kill ? 43 : 41, 1);
   } else if (location == AGX3_VARY_CENTER) {
      ov[n++] = (struct agx3_field_kv)KV(37, 66);
      ov[n++] = (struct agx3_field_kv)KV(39, 7);
      ov[n++] = (struct agx3_field_kv)KV(41, 0);
      ov[n++] = (struct agx3_field_kv)KV(32, 0);
   } else {
      /* AGX3C centroid interpolation uses the same register-source leaf, with
       * the native implicit coordinate register selected by the shade model.
       * G17P has no implicit form: its centroid coordinates are an explicit
       * ITRCOORDS token. */
      if (is_b(enc))
         return false;
      ov[n++] = (struct agx3_field_kv)KV(39, 2);
      ov[n++] = (struct agx3_field_kv)KV(41, 1);
      ov[n++] =
         (struct agx3_field_kv)KV(32, perspective ? 0 : 6);
   }
   if (projective_rhw)
      ov[n++] = (struct agx3_field_kv)KV(54, 1);
   if (first)
      ov[n++] = (struct agx3_field_kv)KV(63, 1);
   return agx3_bind_compose(enc, "f32interp", 0, is_b(enc) ? base_b : base_c,
                            is_b(enc) ? 5 : 7, ov, n, out, len);
}

/* ITRCOORDS modes: f38=2 derives a sample's coordinates from a sample index,
 * f38=1 derives the centroid of the covered samples from a coverage word. */
static bool
agx3_vary_coords(enum agx_isa_encoding enc, uint16_t dest, uint8_t mode,
                 uint16_t source, bool source_kill, uint16_t wait_mask,
                 uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || (dest & 1) != 0 || dest >= 256 ||
       source >= 256 || (wait_mask & ~0x3f) != 0)
      return false;

   /* ITRCOORDS is an eight-byte G17P instruction: register/32-bit
    * destination, register/16-bit source, and the mode in f38. */
   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, 2), KV(11, 1), KV(23, 2), KV(24, 1), KV(38, mode),
   };
   struct agx3_field_kv ov[10];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(16, source);
   if (source_kill)
      ov[n++] = (struct agx3_field_kv)KV(27, 1);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);

   return agx3_bind_compose(enc, "itrcoords", 0, base, 6, ov, n, out, len);
}

bool
agx3_vary_sample_coords(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t sample_index, bool sample_index_kill,
                        uint16_t wait_mask, uint8_t out[AGX3_VARY_MAX_LENGTH],
                        uint8_t *len)
{
   /* Fresh Apple Metal 32023.917 `interpolate_at_sample` and
    * `[[sample_{,no_}perspective]]` witnesses. */
   return agx3_vary_coords(enc, dest, 2, sample_index, sample_index_kill,
                           wait_mask, out, len);
}

bool
agx3_vary_centroid_coords(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t coverage, bool coverage_kill,
                          uint16_t wait_mask,
                          uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   /* Fresh applegpu_g17p `[[centroid_{,no_}perspective]]` witnesses at
    * raster_sample_count 1 and 4 read SR39 and emit ITRCOORDS f38=1 on it,
    * then consume the token exactly as explicit sample interpolation does. */
   return agx3_vary_coords(enc, dest, 1, coverage, coverage_kill, wait_mask,
                           out, len);
}

bool
agx3_vary_interp_sample(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t coef_index, bool perspective, uint16_t coords,
                        bool coords_kill, uint8_t out[AGX3_VARY_MAX_LENGTH],
                        uint8_t *len)
{
   return agx3_vary_interp_mode(enc, dest, coef_index, perspective,
                                AGX3_VARY_CENTER, false, false, true, coords,
                                coords_kill, out, len);
}

bool
agx3_vary_projective_rhw_sample(enum agx_isa_encoding enc, uint16_t dest,
                                uint16_t coef_index, uint16_t coords,
                                bool coords_kill,
                                uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   return agx3_vary_interp_mode(enc, dest, coef_index, true, AGX3_VARY_CENTER,
                                false, true, true, coords, coords_kill, out,
                                len);
}

bool
agx3_vary_interp_at_start(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t coef_index, bool perspective,
                         enum agx3_vary_location location, bool first,
                         uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   return agx3_vary_interp_mode(enc, dest, coef_index, perspective, location,
                                first, false, false, 0, false, out, len);
}

bool
agx3_vary_projective_rhw(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t coef_index, bool first,
                         uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   return agx3_vary_interp_mode(enc, dest, coef_index, true, AGX3_VARY_CENTER,
                                first, true, false, 0, false, out, len);
}

bool
agx3_vary_interp_at(enum agx_isa_encoding enc, uint16_t dest,
                    uint16_t coef_index, bool perspective,
                    enum agx3_vary_location location,
                    uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   return agx3_vary_interp_at_start(enc, dest, coef_index, perspective,
                                    location, false, out, len);
}

bool
agx3_vary_ldcf_at_start(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t coef_index, uint8_t slot, bool first,
                        uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   if ((enc != AGX_ISA_ENCODING_AGX3B &&
        enc != AGX_ISA_ENCODING_AGX3C) ||
       (first && !is_b(enc)) ||
       (dest & 1) != 0 || dest >= 256 || slot < 1 ||
       slot > (is_b(enc) ? 6 : 8))
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 2), KV(10, 2), KV(23, 1),
                                          KV(24, 1), KV(38, 7)};
   const struct agx3_field_kv base_c[] = {KV(9, 2),  KV(10, 2), KV(23, 1),
                                          KV(24, 1), KV(38, 7), KV(29, 103)};
   const struct agx3_field_kv ov[] = {KV(2, dest), KV(15, coef_index),
                                      KV(12, slot), KV(48, first)};
   return agx3_bind_compose(enc, "ldcfr", 0, is_b(enc) ? base_b : base_c,
                            is_b(enc) ? 5 : 6, ov, first ? 4 : 3, out, len);
}

bool
agx3_vary_ldcf(enum agx_isa_encoding enc, uint16_t dest, uint16_t coef_index,
               uint8_t slot, uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   return agx3_vary_ldcf_at_start(enc, dest, coef_index, slot, false, out, len);
}

bool
agx3_vary_uvsw(enum agx_isa_encoding enc, uint16_t value, uint16_t uvs_index,
               uint8_t slot, bool value_kill, uint16_t wait_mask,
               uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   const unsigned nwait = is_b(enc) ? 6 : 8;
   if ((enc != AGX_ISA_ENCODING_AGX3B &&
        enc != AGX_ISA_ENCODING_AGX3C) ||
       (value & 1) != 0 || value >= 256 || slot < 1 || slot > nwait ||
       (wait_mask & ~((1u << nwait) - 1)) != 0)
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 5), KV(23, 1), KV(24, 1)};
   const struct agx3_field_kv base_c[] = {KV(9, 5),  KV(10, 2), KV(23, 1),
                                          KV(24, 1), KV(28, 2), KV(29, 196)};
   struct agx3_field_kv ov[16];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(5, value);
   ov[n++] = (struct agx3_field_kv)KV(15, uvs_index);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(13, value_kill ? 1 : 0);
   /* Wait bits f30.. (6 on agx3b, 8 on agx3c). */
   for (unsigned i = 0; i < nwait; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);

   return agx3_bind_compose(enc, "uvsw", 0, is_b(enc) ? base_b : base_c,
                            is_b(enc) ? 3 : 6, ov, n, out, len);
}

bool
agx3_vary_uvsw_register_index(enum agx_isa_encoding enc, uint16_t value,
                              uint16_t index, uint8_t slot, bool value_kill,
                              uint16_t wait_mask,
                              uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   /* uvsw's index operand has three kinds in Apple's AGX3b encoder table:
    * f23 = 1 immediate (f15), 2 register (f16) and 3 uniform (f17), each a
    * half wide. Metal never computes a vertex output's index, so only the
    * immediate kind is witnessed in compiled programs; the register kind is
    * the table's own form for the same field. */
   if (enc != AGX_ISA_ENCODING_AGX3B || (value & 1) != 0 || value >= 256 ||
       index >= 256 || slot < 1 || slot > 6 || (wait_mask & ~0x3fu) != 0)
      return false;
   const struct agx3_field_kv base[] = {KV(9, 5), KV(23, 2), KV(24, 1)};
   struct agx3_field_kv ov[10];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(5, value);
   ov[n++] = (struct agx3_field_kv)KV(16, index);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(13, value_kill ? 1 : 0);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);

   return agx3_bind_compose(enc, "uvsw", 0, base, 3, ov, n, out, len);
}

bool
agx3_vary_uvsw_uniform_index(enum agx_isa_encoding enc, uint16_t value,
                             uint16_t uniform, uint8_t slot, bool value_kill,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   /* The uniform kind of uvsw's index operand (f23 = 3, half index in f17;
    * see agx3_vary_uvsw_register_index()). */
   if (enc != AGX_ISA_ENCODING_AGX3B || (value & 1) != 0 || value >= 256 ||
       uniform >= 512 || slot < 1 || slot > 6 || (wait_mask & ~0x3fu) != 0)
      return false;
   const struct agx3_field_kv base[] = {KV(9, 5), KV(23, 3), KV(24, 1)};
   struct agx3_field_kv ov[10];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(5, value);
   ov[n++] = (struct agx3_field_kv)KV(17, uniform);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(13, value_kill ? 1 : 0);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);

   return agx3_bind_compose(enc, "uvsw", 0, base, 3, ov, n, out, len);
}
