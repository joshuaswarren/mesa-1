/* SPDX-License-Identifier: MIT */
#include "agx3_uniform_integer.h"
#include <string.h>
#include "agx3_binder.h"

static bool
source_valid(struct agx3_uniform_integer_src s, bool half, bool addend)
{
   if (s.kind == AGX3_UI_ZERO)
      return addend && !s.value && !s.cache && !s.discard;
   if (s.kind != AGX3_UI_REGISTER && s.kind != AGX3_UI_UNIFORM)
      return false;
   if (s.value >=
          (s.kind == AGX3_UI_REGISTER ? AGX3_UNIFORM_INTEGER_GPRS : 512) ||
       (!half && (s.value & 1)) || (s.cache && s.discard))
      return false;
   return s.kind != AGX3_UI_UNIFORM || (!s.cache && !s.discard);
}

bool
agx3_uniform_integer_encode(enum agx_isa_encoding encoding,
                            const struct agx3_uniform_integer_desc *d,
                            uint8_t output[16], uint8_t *length)
{
   if (encoding != AGX_ISA_ENCODING_AGX3B || !d || !output || !length ||
       d->op > AGX3_UI_ANDC || (int)d->op < 0 ||
       d->dest >= AGX3_UNIFORM_INTEGER_GPRS || (!d->half && (d->dest & 1)) ||
       d->wait_mask >= 64 ||
       (d->shift && (d->op != AGX3_UI_ADD || d->shift != 3)))
      return false;

   const bool mad = d->op == AGX3_UI_MAD;
   const bool bitwise = d->op >= AGX3_UI_AND;
   const unsigned count = mad ? 3 : 2, width = d->half ? 1 : 2;
   struct agx3_uniform_integer_src src[3];
   memcpy(src, d->src, sizeof(src));
   for (unsigned i = 0; i < count; ++i)
      if (!source_valid(src[i], d->half, mad && i == 2))
         return false;

   /* Main-program witnesses have one varying multiplicand/binary operand.
    * Canonicalize commutative forms to Apple's uniform-first specimens. */
   if (!((src[0].kind == AGX3_UI_UNIFORM && src[1].kind == AGX3_UI_REGISTER) ||
         (src[0].kind == AGX3_UI_REGISTER && src[1].kind == AGX3_UI_UNIFORM)))
      return false;
   if (src[1].kind == AGX3_UI_UNIFORM && d->op != AGX3_UI_SUB &&
       d->op != AGX3_UI_ANDC) {
      /* A shifted ADD is src0 + (src1 << shift); swapping would change it. */
      if (d->shift)
         return false;
      struct agx3_uniform_integer_src tmp = src[0];
      src[0] = src[1];
      src[1] = tmp;
   }
   if (d->op == AGX3_UI_ANDC && src[0].kind != AGX3_UI_UNIFORM)
      return false;

   const char *mnemonic = NULL;
   unsigned endpoint = 0;
   switch (d->op) {
   case AGX3_UI_ADD:
      mnemonic = "iadd";
      break;
   case AGX3_UI_SUB:
      mnemonic = "isub";
      break;
   case AGX3_UI_MAD:
      mnemonic = "imadd";
      break;
   case AGX3_UI_AND:
      mnemonic = "bit_and";
      endpoint = 1;
      break;
   case AGX3_UI_OR:
      mnemonic = "bit_or";
      endpoint = 1;
      break;
   case AGX3_UI_XOR:
      mnemonic = "bit_xor";
      endpoint = 1;
      break;
   case AGX3_UI_NAND:
      mnemonic = "bit_nand";
      break;
   case AGX3_UI_NOR:
      mnemonic = "bit_nor";
      break;
   case AGX3_UI_ANDC:
      mnemonic = "bit_andc";
      break;
   }

   struct agx3_field_kv fields[32];
   unsigned n = 0;
#define FIELD(f, v) fields[n++] = (struct agx3_field_kv){f, v}
   FIELD(2, d->dest);
   FIELD(9, 2);
   FIELD(10, width);
   FIELD(11, d->dest_cache);
   for (unsigned i = 0; i < count; ++i) {
      /* Integer arithmetic sources are 15 fields apart. Bitwise sources
       * have a 14-field stride after their first operand. */
      unsigned base = 15 + i * (bitwise ? 14 : 15);
      FIELD(base + src[i].kind - 1, src[i].value);
      FIELD(base + 8, src[i].kind);
      FIELD(base + 9, src[i].kind == AGX3_UI_ZERO ? 1 : width);
      if (src[i].kind == AGX3_UI_REGISTER) {
         FIELD(base + 10, src[i].cache);
         FIELD(base + 12, src[i].discard);
      }
   }
   if (d->shift)
      FIELD(54, d->shift);
   for (unsigned slot = 0; slot < 6; ++slot)
      FIELD((mad       ? 60
             : bitwise ? 44
                       : 45) +
               slot,
            !!(d->wait_mask & (1u << slot)));
#undef FIELD
   uint8_t bytes[16], size;
   if (!agx3_bind_compose(encoding, mnemonic, endpoint, NULL, 0, fields, n,
                          bytes, &size))
      return false;
   memcpy(output, bytes, size);
   *length = size;
   return true;
}
