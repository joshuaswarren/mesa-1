/* SPDX-License-Identifier: MIT */
#include "agx3_uniform_alu.h"
#include <string.h>
#include "agx3_binder.h"

static bool
uniform_float_encode(enum agx_isa_encoding encoding,
                     enum agx3_uniform_float_op op, uint16_t dest,
                     const struct agx3_uniform_float_src src[3], bool saturate,
                     bool dest_cache, uint8_t wait_mask, bool half,
                     bool uniform_dest,
                     uint8_t output[16], uint8_t *length)
{
   if (encoding != AGX_ISA_ENCODING_AGX3B || !src || !output || !length ||
       (unsigned)op > AGX3_UNIFORM_FMA || dest >= 256 ||
       (!half && (dest & 1)) || wait_mask > 0x3f)
      return false;

   /* Direct uniform destinations are witnessed in constant programs, with
    * aligned low-file f32 results and plain GPR/immediate sources. Keep the
    * independently qualified main-program source contract unchanged. */
   if (uniform_dest && (half || dest >= 128 || dest_cache || saturate))
      return false;

   const unsigned count = op == AGX3_UNIFORM_FMA ? 3 : 2;
   bool uniform = false, reg = false;
   for (unsigned i = 0; i < count; ++i) {
      if (uniform_dest && (src[i].kind == AGX3_FLOAT_UNIFORM ||
                           src[i].abs || src[i].neg))
         return false;
      switch (src[i].kind) {
      case AGX3_FLOAT_UNIFORM:
         uniform = true;
         if (src[i].value >= 512 || (!half && (src[i].value & 1)) ||
             src[i].abs ||
             (src[i].neg && (half || op != AGX3_UNIFORM_FMA || i != 2)) ||
             src[i].cache || src[i].discard)
            return false;
         break;
      case AGX3_FLOAT_REGISTER:
         reg = true;
         if (src[i].value >= 256 || (!half && (src[i].value & 1)) ||
             (src[i].cache && src[i].discard))
            return false;
         break;
      case AGX3_FLOAT_IMMEDIATE:
         if (src[i].value > 255 || src[i].cache || src[i].discard)
            return false;
         break;
      default:
         return false;
      }
      /* The initial half-uniform corpus grounds plain operands only. */
      if (half && (src[i].abs || src[i].neg || saturate))
         return false;
   }
   /* Native main-program witnesses all retain at least one GPR operand.
    * All-uniform/immediate expressions keep one materialized input until
    * their port and constant-program contracts have their own witness. */
   if (!reg || (!uniform_dest && !uniform))
      return false;

   /* Apple G17P/G16G uniform_repeat: 09010e0d210e02c2. Domain 3 selects
    * uniform f36/f53 instead of register f35/f52. ADD and MUL use the same
    * first two operand records. See probes/agx3-optimization-corpus. */
   const char *mnemonic = half ? (op == AGX3_UNIFORM_FADD   ? "f16add"
                                  : op == AGX3_UNIFORM_FMUL ? "f16mul"
                                                            : "f16fma")
                               : (op == AGX3_UNIFORM_FADD   ? "f32add"
                                  : op == AGX3_UNIFORM_FMUL ? "f32mul"
                                                            : "f32fma");
   uint8_t best[16], best_length = 0;
   for (unsigned swap = 0; swap < 2; ++swap) {
      struct agx3_field_kv fields[48] = {
         {uniform_dest ? 3 : 2, dest},
         {9, uniform_dest ? 3 : 2}, {10, half ? 1 : 2},
         {11, dest_cache}, {14, saturate},
      };
      unsigned n = 5;
      for (unsigned i = 0; i < count; ++i) {
         const struct agx3_uniform_float_src *s = &src[i < 2 ? i ^ swap : i];
         const unsigned base = (half ? 16 : 17) + 17 * i;
         unsigned value = s->value;
         if (s->kind == AGX3_FLOAT_IMMEDIATE) {
            if (s->abs)
               value &= 0x7f;
            if (s->neg)
               value ^= 0x80;
         }
         fields[n++] = (struct agx3_field_kv){base + s->kind - 1, value};
         fields[n++] = (struct agx3_field_kv){base + 8, s->kind};
         fields[n++] = (struct agx3_field_kv){base + 9, half ? 1 : 2};
         fields[n++] = (struct agx3_field_kv){base + 10, s->cache};
         fields[n++] = (struct agx3_field_kv){base + 12, s->discard};
         fields[n++] = (struct agx3_field_kv){
            base + 13, s->kind != AGX3_FLOAT_IMMEDIATE && s->abs};
         fields[n++] = (struct agx3_field_kv){
            base + 14, s->kind != AGX3_FLOAT_IMMEDIATE && s->neg};
      }
      if (!half && op == AGX3_UNIFORM_FMUL) {
         fields[n++] = (struct agx3_field_kv){60, 2};
         fields[n++] = (struct agx3_field_kv){74, 2};
      } else if (!half && op == AGX3_UNIFORM_FMA) {
         fields[n++] = (struct agx3_field_kv){77, 2};
         fields[n++] = (struct agx3_field_kv){91, 2};
      }
      const unsigned wait_base = half ? (op == AGX3_UNIFORM_FMA ? 68 : 51)
                                 : op == AGX3_UNIFORM_FADD ? 52
                                 : op == AGX3_UNIFORM_FMUL ? 80
                                                           : 97;
      for (unsigned i = 0; i < 6; ++i)
         fields[n++] =
            (struct agx3_field_kv){wait_base + i, (wait_mask >> i) & 1};

      const unsigned last_ep = op == AGX3_UNIFORM_FMA ? 5 : 3;
      /* Compact wait selectors express one slot. Use the independent fields
       * for multiple producers, without draining unrelated outstanding work. */
      const unsigned first_ep = (wait_mask & (wait_mask - 1)) ? last_ep : 0;
      for (unsigned ep = first_ep; ep <= last_ep; ++ep) {
         uint8_t bytes[16], size;
         if (agx3_bind_compose(encoding, mnemonic, ep, NULL, 0, fields, n,
                               bytes, &size)) {
            if (!best_length || size < best_length) {
               memcpy(best, bytes, size);
               best_length = size;
            }
            /* These arithmetic endpoints are ordered by increasing length. */
            break;
         }
      }
   }
   if (!best_length)
      return false;
   memcpy(output, best, best_length);
   *length = best_length;
   return true;
}

bool
agx3_uniform_float_encode(enum agx_isa_encoding encoding,
                          enum agx3_uniform_float_op op, uint16_t dest,
                          const struct agx3_uniform_float_src src[3],
                          bool saturate, bool dest_cache, uint8_t wait_mask,
                          uint8_t output[16], uint8_t *length)
{
   return uniform_float_encode(encoding, op, dest, src, saturate, dest_cache,
                               wait_mask, false, false, output, length);
}

bool
agx3_uniform_half_encode(enum agx_isa_encoding encoding,
                         enum agx3_uniform_float_op op, uint16_t dest,
                         const struct agx3_uniform_float_src src[3],
                         bool saturate, bool dest_cache, uint8_t wait_mask,
                         uint8_t output[16], uint8_t *length)
{
   return uniform_float_encode(encoding, op, dest, src, saturate, dest_cache,
                               wait_mask, true, false, output, length);
}

bool
agx3_uniform_float_store_encode(enum agx_isa_encoding encoding,
                                enum agx3_uniform_float_op op, uint16_t dest,
                                const struct agx3_uniform_float_src src[3],
                                uint8_t wait_mask,
                                uint8_t output[16], uint8_t *length)
{
   return uniform_float_encode(encoding, op, dest, src, false, false,
                               wait_mask, false, true, output, length);
}

bool
agx3_uniform_minmax_encode(enum agx_isa_encoding encoding, uint16_t dest,
                           uint16_t reg, uint16_t uniform, bool half,
                           bool maximum, bool cache, bool discard,
                           bool dest_cache, uint8_t wait_mask,
                           uint8_t output[16], uint8_t *length)
{
   if (encoding != AGX_ISA_ENCODING_AGX3B || !output || !length ||
       dest >= 256 || reg >= 256 || uniform >= 512 ||
       (!half && ((dest | reg | uniform) & 1)) || (cache && discard) ||
       wait_mask > 0x3f || (wait_mask & (wait_mask - 1)))
      return false;

   const unsigned width = half ? 1 : 2;
   struct agx3_field_kv fields[32] = {
      {3, dest},        {10, 2},       {11, width},
      {12, dest_cache}, {35, reg},     {42, 2},
      {43, width},      {44, cache},   {46, discard},
      {53, uniform},    {59, 3},       {60, width},
      {69, reg},        {76, 2},       {77, width},
      {78, cache},      {80, discard}, {87, uniform},
      {93, 3},          {94, width},   {110, maximum ? 7 : 3},
   };
   unsigned n = 21;
   for (unsigned i = 0; i < 6; ++i)
      fields[n++] = (struct agx3_field_kv){102 + i, (wait_mask >> i) & 1};
   for (unsigned ep = half ? 5 : 7; ep < 28; ++ep) {
      uint8_t candidate[16], size;
      if (agx3_bind_compose(encoding, "fsel", ep, NULL, 0, fields, n, candidate,
                            &size)) {
         memcpy(output, candidate, size);
         *length = size;
         return true;
      }
   }
   return false;
}
