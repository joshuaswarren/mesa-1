/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <string.h>
#include "agx3_uniform_integer.h"

static unsigned checks, failures;
#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         ++failures;                                                           \
         fprintf(stderr, "FAIL %s:%u: %s\n", __FILE__, __LINE__, #x);          \
      }                                                                        \
   } while (0)

static struct agx3_uniform_integer_desc
desc(enum agx3_uniform_integer_op op, bool half)
{
   struct agx3_uniform_integer_desc d = {
      .op = op,
      .half = half,
      .wait_mask = 32,
      .src = {{AGX3_UI_UNIFORM, 12},
              {AGX3_UI_REGISTER, 0, false, true},
              {AGX3_UI_UNIFORM, half ? 13 : 14}},
   };
   return d;
}

static void
golden(struct agx3_uniform_integer_desc d, const char *hex)
{
   uint8_t bytes[16], size = 0;
   CHECK(agx3_uniform_integer_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &size));
   CHECK(size * 2 == strlen(hex));
   for (unsigned i = 0; i < size && i * 2 < strlen(hex); ++i) {
      unsigned expected = 256;
      CHECK(sscanf(hex + i * 2, "%2x", &expected) == 1);
      CHECK(bytes[i] == expected);
   }
}

static void
rejected(struct agx3_uniform_integer_desc d)
{
   uint8_t bytes[16], size = 0x5a;
   memset(bytes, 0xa5, sizeof(bytes));
   CHECK(
      !agx3_uniform_integer_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &size));
   CHECK(size == 0x5a);
   for (unsigned i = 0; i < sizeof(bytes); ++i)
      CHECK(bytes[i] == 0xa5);
}

int
main(void)
{
   /* Fresh G17P/G16G native first-lane instructions, with slot-six wait and
    * discarded varying operand. Both architectures have identical bytes. */
   static const struct {
      enum agx3_uniform_integer_op op;
      const char *word, *half;
   } cases[] = {
      {AGX3_UI_ADD, "9f015600021800a81704", "9f015600021800001704"},
      {AGX3_UI_SUB, "1f015600021800a81704", "1f015600021800001704"},
      {AGX3_UI_MAD, "9f00560002180070d02f0800", "9f00560002180068002e0800"},
      {AGX3_UI_AND, "0b0d1701800000800000", "030c1700800000800000"},
      {AGX3_UI_OR, "0b0d1701820800800000", "030c1700820800800000"},
      {AGX3_UI_XOR, "0b0d1601820800800000", "030c1600820800800000"},
      {AGX3_UI_NAND, "0b0d1601830800800000", "030c1600830800800000"},
      {AGX3_UI_NOR, "0b0d1601810000800000", "030c1600810000800000"},
      {AGX3_UI_ANDC, "0b0d1601820000800000", "030c1600820000800000"},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      golden(desc(cases[i].op, false), cases[i].word);
      golden(desc(cases[i].op, true), cases[i].half);
   }
   struct agx3_uniform_integer_desc d = desc(AGX3_UI_MAD, false);
   d.src[2] = (struct agx3_uniform_integer_src){AGX3_UI_ZERO, 0};
   golden(d, "9f00560002180000d0260800");
   d.half = true;
   golden(d, "9f0056000218000000260800");
   d = desc(AGX3_UI_MAD, false);
   d.src[1] =
      (struct agx3_uniform_integer_src){AGX3_UI_REGISTER, 6, true, false};
   d.src[2] =
      (struct agx3_uniform_integer_src){AGX3_UI_REGISTER, 12, true, false};
   golden(d, "9f00560002181864d8232800");
   d.half = true;
   d.src[1].value = 4;
   d.src[2].value = 7;
   golden(d, "9f0056000218103c08222800");
   d = desc(AGX3_UI_SUB, false);
   d.src[0] =
      (struct agx3_uniform_integer_src){AGX3_UI_REGISTER, 0, false, true};
   d.src[1] = (struct agx3_uniform_integer_src){AGX3_UI_UNIFORM, 12};
   golden(d, "1f015600020030a81701");
   d.half = true;
   golden(d, "1f015600020030001701");
   d = desc(AGX3_UI_ADD, false);
   d.shift = 3;
   golden(d, "9f015600021800e81614");
   d.half = true;
   golden(d, "9f015600021800401614");
   d = desc(AGX3_UI_NAND, true);
   d.dest = d.src[1].value = 1;
   d.src[0].value = 13;
   d.wait_mask = 0;
   golden(d, "030c16009f0800000000");

   /* Commutative canonicalization preserves the GPR hint as well as the
    * uniform index. SUB/ANDC and shifted ADD are deliberately not commuted. */
   for (unsigned op = AGX3_UI_ADD; op <= AGX3_UI_NOR; ++op) {
      if (op == AGX3_UI_SUB)
         continue;
      for (unsigned half = 0; half < 2; ++half) {
         for (unsigned hint = 0; hint < 3; ++hint) {
            d = desc(op, half);
            d.src[1].discard = hint == 1;
            d.src[1].cache = hint == 2;
            uint8_t a[16], b[16], na = 0, nb = 0;
            CHECK(
               agx3_uniform_integer_encode(AGX_ISA_ENCODING_AGX3B, &d, a, &na));
            struct agx3_uniform_integer_src tmp = d.src[0];
            d.src[0] = d.src[1];
            d.src[1] = tmp;
            CHECK(
               agx3_uniform_integer_encode(AGX_ISA_ENCODING_AGX3B, &d, b, &nb));
            CHECK(na == nb && !memcmp(a, b, na));
         }
      }
   }

   /* Dependency masks must remain distinct through every address boundary.
    * These are structural variations, not additional native hardware tests. */
   const unsigned uniforms[] = {0,   1,   12,  13,  248, 254,
                                255, 256, 504, 510, 511};
   for (unsigned op = AGX3_UI_ADD; op <= AGX3_UI_ANDC; ++op) {
      for (unsigned half = 0; half < 2; ++half) {
         for (unsigned u = 0; u < sizeof(uniforms) / sizeof(uniforms[0]); ++u) {
            if (!half && (uniforms[u] & 1))
               continue;
            d = desc(op, half);
            d.src[0].value = uniforms[u];
            d.src[1].value = d.dest = half ? 191 : 190;
            uint8_t previous[64][16] = {{0}}, sizes[64] = {0};
            for (unsigned w = 0; w < 64; ++w) {
               d.wait_mask = w;
               const bool encoded = agx3_uniform_integer_encode(
                  AGX_ISA_ENCODING_AGX3B, &d, previous[w], &sizes[w]);
               CHECK(encoded);
               if (!encoded) {
                  fprintf(stderr, "boundary op=%u half=%u uniform=%u wait=%u\n",
                          op, half, uniforms[u], w);
                  break;
               }
               for (unsigned before = 0; before < w; ++before)
                  CHECK(sizes[w] != sizes[before] ||
                        memcmp(previous[w], previous[before], sizes[w]));
            }
         }
      }
   }

   d = desc(AGX3_UI_ADD, false);
   d.dest = 1;
   rejected(d);
   d = desc(AGX3_UI_ADD, true);
   d.dest = 256;
   rejected(d);
   d = desc(AGX3_UI_ADD, true);
   d.dest = 192;
   rejected(d);
   d = desc(AGX3_UI_AND, true);
   d.dest = 192;
   rejected(d);
   d = desc(AGX3_UI_ADD, true);
   d.src[0].value = 512;
   rejected(d);
   d = desc(AGX3_UI_ADD, false);
   d.src[0].value = 13;
   rejected(d);
   d = desc(AGX3_UI_ADD, false);
   d.src[1].value = 256;
   rejected(d);
   d = desc(AGX3_UI_AND, true);
   d.src[1].value = 192;
   rejected(d);
   d = desc(AGX3_UI_ADD, false);
   d.src[0].cache = true;
   rejected(d);
   d = desc(AGX3_UI_ADD, false);
   d.src[1].cache = true;
   rejected(d);
   d = desc(AGX3_UI_ADD, false);
   d.wait_mask = 64;
   rejected(d);
   d = desc(AGX3_UI_ADD, false);
   d.shift = 1;
   rejected(d);
   d = desc(AGX3_UI_ADD, false);
   d.src[1].kind = AGX3_UI_UNIFORM;
   rejected(d);
   d = desc(AGX3_UI_ADD, false);
   d.src[0].kind = AGX3_UI_REGISTER;
   rejected(d);
   d = desc(AGX3_UI_MAD, false);
   d.src[2].kind = AGX3_UI_ZERO;
   rejected(d);
   d = desc(AGX3_UI_ADD, false);
   d.src[0] = (struct agx3_uniform_integer_src){AGX3_UI_REGISTER, 0};
   d.src[1] = (struct agx3_uniform_integer_src){AGX3_UI_UNIFORM, 12};
   d.shift = 3;
   rejected(d);
   d.op = AGX3_UI_ANDC;
   d.shift = 0;
   rejected(d);
   d = desc(AGX3_UI_ADD, false);
   uint8_t output[16], length;
   CHECK(
      !agx3_uniform_integer_encode(AGX_ISA_ENCODING_AGX2, &d, output, &length));
   CHECK(!agx3_uniform_integer_encode(AGX_ISA_ENCODING_AGX3C, &d, output,
                                      &length));
   CHECK(!agx3_uniform_integer_encode(AGX_ISA_ENCODING_AGX3B, NULL, output,
                                      &length));
   CHECK(
      !agx3_uniform_integer_encode(AGX_ISA_ENCODING_AGX3B, &d, NULL, &length));
   CHECK(
      !agx3_uniform_integer_encode(AGX_ISA_ENCODING_AGX3B, &d, output, NULL));
   printf("integer uniform probe: %u checks, %u failures\n", checks, failures);
   return failures != 0;
}
