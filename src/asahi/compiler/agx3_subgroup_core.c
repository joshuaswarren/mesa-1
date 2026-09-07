/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_subgroup_core.h"

#include "agx3_binder.h"

#define ADD_FIELD(F, V)                                                        \
   do {                                                                        \
      base[nb++] = (struct agx3_field_kv){(F), (V)};                           \
   } while (0)

static bool
is_c(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3C;
}

static bool
valid_reg(enum agx_isa_encoding enc, uint16_t reg)
{
   const uint16_t limit = enc == AGX_ISA_ENCODING_AGX3B   ? 192
                          : enc == AGX_ISA_ENCODING_AGX3C ? 256
                                                          : 0;
   return limit != 0 && (reg & 1) == 0 && reg < limit;
}

static const char *
shuffle_mnemonic(bool quad, enum agx3_subgroup_shuffle_op op)
{
   static const char *const simd[] = {
      "simdbcast",
      "simdbfly",
      "simdshftup",
      "simdshftdn",
   };
   static const char *const q[] = {
      "quadshfl",
      "quadbfly",
      "quadshftup",
      "quadshftdn",
   };
   return (unsigned)op < AGX3_SUBGROUP_SHUFFLE_NUM
             ? (quad ? q[op] : simd[op])
             : NULL;
}

bool
agx3_subgroup_encode_shuffle(enum agx_isa_encoding enc,
                             const struct agx3_subgroup_shuffle_desc *d,
                             uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                             uint8_t *length)
{
   if (d == NULL || out == NULL || length == NULL || !valid_reg(enc, d->dest) ||
       !valid_reg(enc, d->src))
      return false;

   const char *mnemonic = shuffle_mnemonic(d->quad, d->op);
   if (mnemonic == NULL)
      return false;

   /*
    * The immediate-lane form, added from the g17p session. f37 = 1 selects it
    * and the lane index sits in f29; f39 and f41, which the register form
    * carries, are absent. The per-slot wait bits are f44 + (slot - 1): PROVEN
    * by three probes whose source comes from a device load on slot 6, all of
    * which set f49 and nothing else.
    */
   if (d->lane_kind == AGX3_SUBGROUP_LANE_IMMEDIATE) {
      const unsigned slots = 6;
      /* G17P witnesses the SIMD broadcast subset and the exact quad family
       * used by the public GLSL operations below. Keep every other immediate
       * shape fail-closed. */
      const bool simd_witness =
         !d->quad && d->op == AGX3_SUBGROUP_SHUFFLE_BROADCAST &&
         (d->lane == 0 || d->lane == 3 || d->lane == 5);
      const bool quad_witness =
         d->quad && ((d->op == AGX3_SUBGROUP_SHUFFLE_BROADCAST && d->lane == 2) ||
                     (d->op == AGX3_SUBGROUP_SHUFFLE_XOR &&
                      d->lane >= 1 && d->lane <= 3));
      if (enc != AGX_ISA_ENCODING_AGX3B || (!simd_witness && !quad_witness))
         return false;
      if (d->quad && (d->dest_hint ||
                      (d->op == AGX3_SUBGROUP_SHUFFLE_XOR &&
                       d->wait_mask != 0)))
         return false;
      if (d->lane > 0x7f || (d->wait_mask >> slots) != 0)
         return false;

      struct agx3_field_kv ibase[8];
      unsigned inb = 0;
      ibase[inb++] = (struct agx3_field_kv){9, 2};
      ibase[inb++] = (struct agx3_field_kv){10, 2};
      ibase[inb++] = (struct agx3_field_kv){23, 2};
      ibase[inb++] = (struct agx3_field_kv){24, 2};
      if (d->quad) {
         ibase[inb++] = (struct agx3_field_kv){d->src_live ? 25 : 27, 1};
      } else {
         ibase[inb++] = (struct agx3_field_kv){27, 1};
      }
      ibase[inb++] = (struct agx3_field_kv){37, 1};
      ibase[inb++] = (struct agx3_field_kv){38, 1};

      struct agx3_field_kv iov[14];
      unsigned inv = 0;
      iov[inv++] = (struct agx3_field_kv){2, d->dest};
      iov[inv++] = (struct agx3_field_kv){11, d->dest_hint ? 1u : 0u};
      iov[inv++] = (struct agx3_field_kv){16, d->src};
      iov[inv++] = (struct agx3_field_kv){29, d->lane};
      for (unsigned i = 0; i < slots; ++i)
         if (d->wait_mask & (1u << i))
            iov[inv++] = (struct agx3_field_kv){(uint16_t)(44 + i), 1};

      return agx3_bind_compose(enc, mnemonic, 0, ibase, inb, iov, inv, out,
                               length);
   }

   if (d->wait_mask != 0 ||
       (d->lane_kind != AGX3_SUBGROUP_LANE_REGISTER &&
        d->lane_kind != AGX3_SUBGROUP_LANE_UNIFORM) ||
       (d->lane & 1) != 0 ||
       (d->lane_kind == AGX3_SUBGROUP_LANE_REGISTER ? !valid_reg(enc, d->lane)
                                                    : d->lane >= 512))
      return false;

   struct agx3_field_kv base[16];
   unsigned nb = 0;
   ADD_FIELD(9, 2);
   ADD_FIELD(10, 2);
   ADD_FIELD(23, 2);
   ADD_FIELD(24, 2);
   ADD_FIELD(37, d->lane_kind);
   ADD_FIELD(38, 1);

   if (!is_c(enc)) {
      if (d->lane_kind == AGX3_SUBGROUP_LANE_REGISTER) {
         if (!d->quad && d->op == AGX3_SUBGROUP_SHUFFLE_BROADCAST) {
            ADD_FIELD(11, 1);
            ADD_FIELD(25, 1);
            ADD_FIELD(41, 1);
         } else if (d->op == AGX3_SUBGROUP_SHUFFLE_UP ||
                    d->op == AGX3_SUBGROUP_SHUFFLE_DOWN) {
            ADD_FIELD(11, 1);
            ADD_FIELD(25, 1);
            ADD_FIELD(39, 1);
         } else {
            ADD_FIELD(27, 1);
            ADD_FIELD(41, 1);
         }
      } else if (d->op == AGX3_SUBGROUP_SHUFFLE_UP ||
                 d->op == AGX3_SUBGROUP_SHUFFLE_DOWN) {
         ADD_FIELD(11, 1);
         ADD_FIELD(25, 1);
      } else {
         ADD_FIELD(27, 1);
      }
   } else {
      if (d->lane_kind == AGX3_SUBGROUP_LANE_REGISTER && !d->quad &&
          d->op == AGX3_SUBGROUP_SHUFFLE_BROADCAST) {
         ADD_FIELD(11, 1);
         ADD_FIELD(25, 1);
      } else {
         ADD_FIELD(27, 1);
      }
      if (d->lane_kind == AGX3_SUBGROUP_LANE_REGISTER)
         ADD_FIELD(41, 1);
      static const uint8_t ids[2][AGX3_SUBGROUP_SHUFFLE_NUM] = {
         {154, 155, 161, 160},
         {140, 137, 142, 141},
      };
      ADD_FIELD(42, 3);
      ADD_FIELD(43, ids[d->quad][d->op]);
   }

   struct agx3_field_kv ov[] = {
      {2, d->dest},
      {16, d->src},
      {d->lane_kind == AGX3_SUBGROUP_LANE_REGISTER ? 30 : 31, d->lane},
   };
   return agx3_bind_compose(enc, mnemonic, 0, base, nb, ov, 3, out, length);
}

bool
agx3_subgroup_encode_ballot(enum agx_isa_encoding enc,
                            const struct agx3_subgroup_ballot_desc *d,
                            uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                            uint8_t *length)
{
   if (d == NULL || out == NULL || length == NULL || d->wait_mask != 0 ||
       !valid_reg(enc, d->dest) || !valid_reg(enc, d->src) ||
       (unsigned)d->type > AGX3_SUBGROUP_BALLOT_FLOAT)
      return false;

   if (d->type == AGX3_SUBGROUP_BALLOT_INTEGER) {
      if (d->condition != 4 && d->condition != 5 && d->condition != 6)
         return false;
   } else if (d->condition != 0 && d->condition != 1 && d->condition != 2 &&
              d->condition != 5 && d->condition != 6) {
      return false;
   }

   struct agx3_field_kv base[16];
   unsigned nb = 0;
   ADD_FIELD(9, 2);
   ADD_FIELD(10, 2);
   if (d->quad)
      ADD_FIELD(11, 1);
   ADD_FIELD(23, 2);
   ADD_FIELD(24, 2);
   ADD_FIELD(27, 1);

   const char *mnemonic;
   uint16_t cond_field, invert_field;
   if (d->type == AGX3_SUBGROUP_BALLOT_INTEGER) {
      mnemonic = d->quad ? "quadiballot" : "simdiballot";
      if (d->condition == 4) {
         ADD_FIELD(38, 2);
         ADD_FIELD(39, 2);
         ADD_FIELD(42, 1);
      } else {
         ADD_FIELD(38, 1);
         ADD_FIELD(39, 1);
      }
      cond_field = 54;
      invert_field = 55;
      if (is_c(enc)) {
         ADD_FIELD(44, 3);
         ADD_FIELD(45, d->quad ? 139 : 157);
      }
   } else {
      mnemonic = d->quad ? "quadfballot" : "simdfballot";
      ADD_FIELD(40, 1);
      ADD_FIELD(41, 2);
      cond_field = 58;
      invert_field = 59;
      if (is_c(enc)) {
         ADD_FIELD(48, 3);
         ADD_FIELD(49, d->quad ? 138 : 156);
      }
   }

   struct agx3_field_kv ov[] = {
      {2, d->dest},
      {16, d->src},
      {cond_field, d->condition},
      {invert_field, d->invert},
   };
   return agx3_bind_compose(enc, mnemonic, 0, base, nb, ov, 4, out, length);
}

struct reduce_form {
   const char *simd;
   const char *quad;
   uint8_t c_simd;
   uint8_t c_quad;
   bool is_float;
   uint8_t integer_class;
};

static const struct reduce_form reduce_forms[AGX3_SUBGROUP_REDUCE_NUM] = {
   [AGX3_SUBGROUP_REDUCE_AND] = {"sRedAND", "", 169, 0, false, 2},
   [AGX3_SUBGROUP_REDUCE_FADD] = {"sRedf32add", "qRedf32add", 172, 126, true,
                                  0},
   [AGX3_SUBGROUP_REDUCE_OR] = {"sRedOR", "", 177, 0, false, 2},
   [AGX3_SUBGROUP_REDUCE_FMUL] = {"sRedf32mul", "qRedf32mul", 173, 127, true,
                                  0},
   [AGX3_SUBGROUP_REDUCE_XOR] = {"sRedXOR", "", 182, 0, false, 2},
   [AGX3_SUBGROUP_REDUCE_FMIN] = {"sRedfmin", "qRedfmin", 175, 129, true, 0},
   [AGX3_SUBGROUP_REDUCE_FMAX] = {"sRedfmax", "qRedfmax", 174, 128, true, 0},
   [AGX3_SUBGROUP_REDUCE_IADD] = {"sRediadd", "qRediadd", 176, 130, false, 0},
   [AGX3_SUBGROUP_REDUCE_SMIN] = {"sRedsmin", "qRedsmin", 179, 0, false, 1},
   [AGX3_SUBGROUP_REDUCE_SMAX] = {"sRedsmax", "qRedsmax", 178, 0, false, 1},
   [AGX3_SUBGROUP_REDUCE_UMIN] = {"sRedumin", "qRedumin", 181, 0, false, 2},
   [AGX3_SUBGROUP_REDUCE_UMAX] = {"sRedumax", "qRedumax", 180, 0, false, 2},
};

bool
agx3_subgroup_encode_reduce(enum agx_isa_encoding enc,
                            const struct agx3_subgroup_reduce_desc *d,
                            uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                            uint8_t *length)
{
   if (d == NULL || out == NULL || length == NULL ||
       (unsigned)d->op >= AGX3_SUBGROUP_REDUCE_NUM ||
       !valid_reg(enc, d->dest) ||
       !valid_reg(enc, d->src))
      return false;

   const struct reduce_form *f = &reduce_forms[d->op];
   const char *mnemonic = d->quad ? f->quad : f->simd;
   if (mnemonic == NULL || mnemonic[0] == '\0')
      return false;
   /* The newly recovered integer min/max quad words are G17P/AGX3b
    * witnesses. Preserve the existing AGX3c boundary until independently
    * captured on that encoding. */
   if (d->quad && is_c(enc) && d->op >= AGX3_SUBGROUP_REDUCE_SMIN)
      return false;
   if (d->prefix && (d->quad || (d->op != AGX3_SUBGROUP_REDUCE_FADD &&
                                 d->op != AGX3_SUBGROUP_REDUCE_FMUL &&
                                 d->op != AGX3_SUBGROUP_REDUCE_IADD)))
      return false;

   struct agx3_field_kv base[16];
   unsigned nb = 0;
   ADD_FIELD(9, 2);
   ADD_FIELD(10, 2);
   if (f->is_float) {
      ADD_FIELD(25, 2);
      ADD_FIELD(26, 2);
      if (!d->src_live)
         ADD_FIELD(29, 1);
      if (is_c(enc)) {
         ADD_FIELD(47, 3);
         ADD_FIELD(48, d->quad ? f->c_quad : f->c_simd);
      } else {
         ADD_FIELD(34, 1);
         ADD_FIELD(42, 1);
         ADD_FIELD(43, 1);
      }
   } else {
      ADD_FIELD(23, 2);
      ADD_FIELD(24, 2);
      ADD_FIELD(!is_c(enc) && d->quad && d->src_live ? 25 : 27, 1);
      if (f->integer_class == 1)
         ADD_FIELD(28, 1);
      if (is_c(enc)) {
         if (f->integer_class == 2) {
            ADD_FIELD(42, 3);
            ADD_FIELD(43, d->quad ? f->c_quad : f->c_simd);
         } else {
            ADD_FIELD(43, 3);
            ADD_FIELD(44, d->quad ? f->c_quad : f->c_simd);
         }
      } else if (f->integer_class == 2) {
         ADD_FIELD(29, 1);
         ADD_FIELD(37, 1);
         ADD_FIELD(38, 1);
      } else {
         ADD_FIELD(30, 1);
         ADD_FIELD(38, 1);
         ADD_FIELD(39, 1);
      }
   }

   if (d->prefix)
      ADD_FIELD(f->is_float ? 57 : 53, 1);

   /*
    * Per-slot wait bits. Their base differs by operand family, each pinned by
    * a g17p probe whose source arrives on scoreboard slot 6:
    *   float reductions            f49 + (slot - 1)   (sRedf32add, sRedfmax)
    *   integer add / signed min-max
    *                               f45 + (slot - 1)   (sRediadd, sRedimin,
    *                                                   sRedimax; also catomic
    *                                                   at slot 1)
    *   bitwise / unsigned min-max  f44 + (slot - 1)   (sRedAND, sRedumax)
    * Own-shader A18 Pro captures pin slot-6 signed min/max to this family:
    *   sRedimin 3f02560002001407, sRedimax bf02560002001407.
    */
   const unsigned slots = is_c(enc) ? 8 : 6;
   if (d->wait_mask != 0) {
      if (is_c(enc) || (d->wait_mask >> slots) != 0)
         return false;
      uint16_t wait_base;
      if (f->is_float)
         wait_base = 49;
      else if (f->integer_class == 2)
         wait_base = 44;
      else if (f->integer_class == 0 || f->integer_class == 1)
         wait_base = 45;
      else
         return false;
      for (unsigned i = 0; i < slots; ++i)
         if (d->wait_mask & (1u << i))
            ADD_FIELD((uint16_t)(wait_base + i), 1);
   }

   struct agx3_field_kv ov[4];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv){2, d->dest};
   ov[n++] = (struct agx3_field_kv){f->is_float ? 18u : 16u, d->src};
   ov[n++] = (struct agx3_field_kv){11, d->dest_hint ? 1u : 0u};
   /* Float and G17P quad integer reductions pick between their witnessed
    * source-liveness bits. Other integer families retain the pinned f27. */
   if (f->is_float && d->src_live)
      ov[n++] = (struct agx3_field_kv){27, 1};
   return agx3_bind_compose(enc, mnemonic, 0, base, nb, ov, n, out, length);
}

#undef ADD_FIELD
