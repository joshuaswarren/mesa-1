/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Every positive vector below is one instruction Apple's own offline
 * translator produced. A ladder of MSL kernels
 *
 *     o[i] = (a[i] OP b[i]) ? c[i] : d[i];
 *
 * was compiled with `metal-nt -arch applegpu_g16g` (AGX3b, G17P's encoding),
 * once per comparison operator, plus immediate-select and boolean variants.
 * The condition field tracked OP one-for-one and is numbered exactly like
 * enum agx_icond. Apple has no le/ge/ne condition: it emits the inverse
 * condition with the selected operands swapped, which the ladder shows
 * directly (ule reuses ugt, uge reuses ult, une reuses eq, and so on).
 */

#include "agx3_isel_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;
static unsigned failures;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);          \
         ++failures;                                                           \
      }                                                                        \
   } while (0)

struct isel_vec {
   const char *name;
   struct agx3_isel_desc d;
   uint8_t bytes[AGX3_ISEL_MAX_LENGTH];
   uint8_t length;
};

static const struct isel_vec vectors[] = {
   /* Apple's `isel_chain` keeps this 16-bit boolean result live for the
    * following select. f12 is the only non-register field that differs from
    * the direct-store twin, establishing it as destination cache. */
   {"u16_result_cached", {.dest_16=true, .cmp1_immediate=true,
       .dest_cache=true, .cmp0=0, .cmp1=1, .cond=AGX3_ISEL_ICOND_EQ,
       .true_kind=AGX3_ISEL_IMMEDIATE, .false_kind=AGX3_ISEL_IMMEDIATE,
       .sel_true=1, .sel_false=0, .discard={true,false,false,false},
       .wait_mask=32},
    {0x02,0x01,0x2d,0x80,0x26,0x81,0x17,0xc2,0x20,0x80}, 10},
   {"s32_zero", {.dest_16=false, .cmp_16=false,
       .cmp1_immediate=true, .explicit_hints=true, .cmp0=0, .cmp1=0,
       .cond=AGX3_ISEL_ICOND_SLT, .sel_true=2, .sel_false=4,
       .discard={true,false,true,true}, .wait_mask=32},
    {0x02,0x01,0x0f,0x80,0x82,0x02,0x07,0xc2,0x80,0x04}, 10},
   {"s32_u8", {.dest_16=false, .cmp_16=false,
       .cmp1_immediate=true, .explicit_hints=true, .cmp0=0, .cmp1=255,
       .cond=AGX3_ISEL_ICOND_SGT, .sel_true=2, .sel_false=4,
       .discard={true,false,true,true}, .wait_mask=32},
    {0x02,0x01,0x1f,0xfe,0x82,0x02,0x16,0xc2,0x80,0x04}, 10},
   {"s16_u8", {.dest_16=true, .cmp_16=true,
       .cmp1_immediate=true, .explicit_hints=true, .cmp0=0, .cmp1=2,
       .cond=AGX3_ISEL_ICOND_SLT, .sel_true=1, .sel_false=2,
       .discard={true,false,true,true}, .wait_mask=32},
    {0x02,0x00,0x0d,0x82,0x82,0x01,0x07,0xc2,0x82,0x02}, 10},
   {"s16_regs", {.dest_16=true, .cmp_16=true,
       .cmp1_immediate=false, .explicit_hints=true, .cmp0=0, .cmp1=1,
       .cond=AGX3_ISEL_ICOND_SGT, .sel_true=2, .sel_false=3,
       .discard={true,true,true,true}, .wait_mask=32},
    {0x02,0x00,0x1d,0x00,0x82,0x02,0x16,0xc8,0x82,0x03}, 10},
   {"u32_mixed_cached_result", {
       .cmp1_immediate=true, .cmp1=2, .cond=AGX3_ISEL_ICOND_EQ,
       .dest_cache=true,
       .discard={true,false,true,false}, .false_kind=AGX3_ISEL_IMMEDIATE,
       .sel_true=2, .sel_false=13, .wait_mask=32},
    {0x02,0x01,0x2f,0x82,0x86,0x02,0x07,0xc2,0x20,0x8d}, 10},
   {"u32_zero_dead", {.wait_mask=32,.cmp1_immediate=true,.cond=AGX3_ISEL_ICOND_EQ,.sel_true=2,.sel_false=4,.discard={1,0,1,1}},
    {0x02,0x01,0x0f,0x80,0x86,0x02,0x07,0xc2,0x80,0x04}, 10},
   {"u32_regs_dead", {.wait_mask=32,.explicit_hints=true,.cmp1=2,.cond=AGX3_ISEL_ICOND_ULT,.sel_true=4,.sel_false=6,.discard={1,1,1,1}},
    {0x02,0x01,0x1f,0x03,0x82,0x04,0x05,0xc0,0x80,0x06}, 10},
   {"u32_regs_cached", {.wait_mask=32,.explicit_hints=true,.cmp0=2,.cmp1=4,.cond=AGX3_ISEL_ICOND_ULT,.sel_true=6,.sel_false=8,.cache={1,1,1,1}},
    {0x02,0x83,0x07,0x85,0x02,0x86,0x05,0xc0,0x00,0x88}, 10},
   {"u16_zero_dead", {.wait_mask=32,.cmp_16=true,.dest_16=true,.cmp1_immediate=true,.cond=AGX3_ISEL_ICOND_EQ,.sel_true=1,.sel_false=2,.discard={1,0,1,1}},
    {0x02,0x00,0x0d,0x80,0x86,0x01,0x07,0xc2,0x82,0x02}, 10},
   {"u16_regs_dead", {.wait_mask=32,.cmp_16=true,.dest_16=true,.cmp1=1,.cond=AGX3_ISEL_ICOND_ULT,.sel_true=2,.sel_false=3,.discard={1,1,1,1}},
    {0x02,0x00,0x1d,0x00,0x82,0x02,0x15,0xc0,0x80,0x03}, 10},
   {"u16_regs_cached", {.wait_mask=32,.cmp_16=true,.dest_16=true,.cmp0=2,.cmp1=3,.cond=AGX3_ISEL_ICOND_EQ,.sel_true=4,.sel_false=5,.cache={1,1,1,1}},
    {0x02,0x82,0x05,0x82,0x06,0x84,0x17,0xc8,0x02,0x85}, 10},
   /* `(a OP b) ? c : d` -- dest r0, compares halves 2 and 6, selects 12/16. */
   {"ult", {.dest = 0, .cmp0 = 2, .cmp1 = 6, .cond = AGX3_ISEL_ICOND_ULT,
            .sel_true = 12, .sel_false = 16, .wait_mask = 1u << 5},
    {0x02, 0x83, 0x07, 0x87, 0x22, 0x0c, 0x05, 0xc0, 0x20, 0x10}, 10},
   {"ugt", {.dest = 0, .cmp0 = 2, .cmp1 = 6, .cond = AGX3_ISEL_ICOND_UGT,
            .sel_true = 12, .sel_false = 16, .wait_mask = 1u << 5},
    {0x02, 0x83, 0x07, 0x87, 0x22, 0x0c, 0x04, 0xc0, 0x20, 0x10}, 10},
   {"ule", {.dest = 0, .cmp0 = 2, .cmp1 = 6, .cond = AGX3_ISEL_ICOND_UGT,
            .sel_true = 16, .sel_false = 12, .wait_mask = 1u << 5},
    {0x02, 0x83, 0x07, 0x87, 0x22, 0x10, 0x04, 0xc0, 0x20, 0x0c}, 10},
   {"uge", {.dest = 0, .cmp0 = 2, .cmp1 = 6, .cond = AGX3_ISEL_ICOND_ULT,
            .sel_true = 16, .sel_false = 12, .wait_mask = 1u << 5},
    {0x02, 0x83, 0x07, 0x87, 0x22, 0x10, 0x05, 0xc0, 0x20, 0x0c}, 10},
   {"ueq", {.dest = 0, .cmp0 = 2, .cmp1 = 6, .cond = AGX3_ISEL_ICOND_EQ,
            .sel_true = 12, .sel_false = 16, .wait_mask = 1u << 5},
    {0x02, 0x83, 0x07, 0x87, 0x26, 0x0c, 0x07, 0xc0, 0x20, 0x10}, 10},
   {"une", {.dest = 0, .cmp0 = 2, .cmp1 = 6, .cond = AGX3_ISEL_ICOND_EQ,
            .sel_true = 16, .sel_false = 12, .wait_mask = 1u << 5},
    {0x02, 0x83, 0x07, 0x87, 0x26, 0x10, 0x07, 0xc0, 0x20, 0x0c}, 10},
   {"slt", {.dest = 0, .cmp0 = 2, .cmp1 = 6, .cond = AGX3_ISEL_ICOND_SLT,
            .sel_true = 12, .sel_false = 16, .wait_mask = 1u << 5},
    {0x02, 0x83, 0x07, 0x87, 0x22, 0x0c, 0x07, 0xc0, 0x20, 0x10}, 10},
   {"sgt", {.dest = 0, .cmp0 = 2, .cmp1 = 6, .cond = AGX3_ISEL_ICOND_SGT,
            .sel_true = 12, .sel_false = 16, .wait_mask = 1u << 5},
    {0x02, 0x83, 0x07, 0x87, 0x22, 0x0c, 0x06, 0xc0, 0x20, 0x10}, 10},
   {"sle", {.dest = 0, .cmp0 = 2, .cmp1 = 6, .cond = AGX3_ISEL_ICOND_SGT,
            .sel_true = 16, .sel_false = 12, .wait_mask = 1u << 5},
    {0x02, 0x83, 0x07, 0x87, 0x22, 0x10, 0x06, 0xc0, 0x20, 0x0c}, 10},
   {"sge", {.dest = 0, .cmp0 = 2, .cmp1 = 6, .cond = AGX3_ISEL_ICOND_SLT,
            .sel_true = 16, .sel_false = 12, .wait_mask = 1u << 5},
    {0x02, 0x83, 0x07, 0x87, 0x22, 0x10, 0x07, 0xc0, 0x20, 0x0c}, 10},

   /* `(a < b) ? 5u : 3u` and `? 255u : 9u` -- 8-bit immediates, 32-bit dest. */
   {"imm5_3", {.dest = 0, .cmp0 = 0, .cmp1 = 4, .cond = AGX3_ISEL_ICOND_ULT,
               .true_kind = AGX3_ISEL_IMMEDIATE, .false_kind = AGX3_ISEL_IMMEDIATE, .sel_true = 5, .sel_false = 3, .wait_mask = 1u << 5},
    {0x02, 0x01, 0x1f, 0x05, 0x22, 0x85, 0x05, 0xc0, 0x20, 0x83}, 10},
   {"imm255_9", {.dest = 0, .cmp0 = 0, .cmp1 = 4, .cond = AGX3_ISEL_ICOND_ULT,
                 .true_kind = AGX3_ISEL_IMMEDIATE, .false_kind = AGX3_ISEL_IMMEDIATE, .sel_true = 255, .sel_false = 9, .wait_mask = 1u << 5},
    {0x02, 0x01, 0x1f, 0x05, 0xa2, 0xff, 0x05, 0xc0, 0x20, 0x89}, 10},

   /* Mixed: register when true / immediate when false, and the mirror. */
   {"reg_imm", {.dest = 0, .cmp0 = 0, .cmp1 = 8, .cond = AGX3_ISEL_ICOND_ULT,
                .true_kind = AGX3_ISEL_REGISTER,
                .false_kind = AGX3_ISEL_IMMEDIATE,
                .sel_true = 10, .sel_false = 9},
    {0x02, 0x01, 0x3f, 0x09, 0x82, 0x0a, 0x05, 0x00, 0x20, 0x89}, 10},
   {"imm_reg", {.dest = 0, .cmp0 = 0, .cmp1 = 8, .cond = AGX3_ISEL_ICOND_ULT,
                .true_kind = AGX3_ISEL_IMMEDIATE,
                .false_kind = AGX3_ISEL_REGISTER,
                .sel_true = 9, .sel_false = 10},
    {0x02, 0x01, 0x3f, 0x09, 0x22, 0x89, 0x05, 0x00, 0x80, 0x0a}, 10},

   /* 16-bit results: the three non-boolean shapes, all at ep25. */
   {"h_reg_imm", {.dest = 2, .dest_16 = true, .cmp0 = 2, .cmp1 = 4,
                  .cond = AGX3_ISEL_ICOND_ULT,
                  .true_kind = AGX3_ISEL_REGISTER,
                  .false_kind = AGX3_ISEL_IMMEDIATE,
                  .sel_true = 6, .sel_false = 7, .wait_mask = 1u << 5},
    {0x12, 0x03, 0x1d, 0x05, 0x82, 0x06, 0x05, 0xc0, 0x20, 0x87}, 10},
   {"h_imm_reg", {.dest = 2, .dest_16 = true, .cmp0 = 2, .cmp1 = 4,
                  .cond = AGX3_ISEL_ICOND_ULT,
                  .true_kind = AGX3_ISEL_IMMEDIATE,
                  .false_kind = AGX3_ISEL_REGISTER,
                  .sel_true = 7, .sel_false = 6, .wait_mask = 1u << 5},
    {0x12, 0x03, 0x1d, 0x05, 0x22, 0x87, 0x05, 0xc0, 0x80, 0x06}, 10},
   {"h_reg_reg", {.dest = 2, .dest_16 = true, .cmp0 = 2, .cmp1 = 4,
                  .cond = AGX3_ISEL_ICOND_ULT,
                  .true_kind = AGX3_ISEL_REGISTER,
                  .false_kind = AGX3_ISEL_REGISTER,
                  .sel_true = 7, .sel_false = 6, .wait_mask = 1u << 5},
    {0x12, 0x03, 0x1d, 0x05, 0x82, 0x07, 0x05, 0xc0, 0x80, 0x06}, 10},

   /* Odd 16-bit destinations, which are ep26 rather than ep25. */
   {"h_odd_bool", {.dest = 3, .dest_16 = true, .cmp0 = 6, .cmp1 = 4,
                   .cond = AGX3_ISEL_ICOND_ULT,
                   .true_kind = AGX3_ISEL_IMMEDIATE,
                   .false_kind = AGX3_ISEL_IMMEDIATE,
                   .sel_true = 1, .sel_false = 0},
    {0x1a, 0x07, 0x1d, 0x05, 0x22, 0x81, 0x05, 0x00, 0x20, 0x80}, 10},
   {"h_odd_reg_imm", {.dest = 3, .dest_16 = true, .cmp0 = 8, .cmp1 = 6,
                      .cond = AGX3_ISEL_ICOND_ULT,
                      .true_kind = AGX3_ISEL_REGISTER,
                      .false_kind = AGX3_ISEL_IMMEDIATE,
                      .sel_true = 10, .sel_false = 5, .wait_mask = 1u << 2},
    {0x1a, 0x09, 0x1d, 0x07, 0x82, 0x0a, 0x05, 0x60, 0x20, 0x85}, 10},

   /* Uniform operands, metal-nt 32023.850 -arch applegpu_g17p kernels:
    * (u.x < r) ? r : r, (r < u.x) ? r : r, (r == 7) ? u.y : r,
    * (r == 7) ? r : u.z, (r == 7) ? u.y : u.z, (r < r) ? u.y : 3u. */
   {"u_cmp0", {.dest = 0, .cmp0 = 8, .cmp0_uniform = true, .cmp1 = 0,
               .cond = AGX3_ISEL_ICOND_ULT, .explicit_hints = true,
               .discard = {false, true, false, false},
               .true_kind = AGX3_ISEL_REGISTER,
               .false_kind = AGX3_ISEL_REGISTER,
               .sel_true = 2, .sel_false = 4, .wait_mask = 1u << 5},
    {0x02, 0x09, 0x17, 0x01, 0x02, 0x02, 0x85, 0xc0, 0x00, 0x04}, 10},
   {"u_cmp1", {.dest = 0, .cmp0 = 0, .cmp1 = 8, .cmp1_uniform = true,
               .cond = AGX3_ISEL_ICOND_ULT, .explicit_hints = true,
               .discard = {true, false, false, false},
               .true_kind = AGX3_ISEL_REGISTER,
               .false_kind = AGX3_ISEL_REGISTER,
               .sel_true = 2, .sel_false = 4, .wait_mask = 1u << 5},
    {0x02, 0x01, 0x0f, 0x09, 0x02, 0x02, 0x05, 0xc2, 0x00, 0x04}, 10},
   {"u_true", {.dest = 0, .cmp0 = 0, .cmp1 = 7, .cmp1_immediate = true,
               .cond = AGX3_ISEL_ICOND_EQ, .explicit_hints = true,
               .discard = {true, false, false, false},
               .true_kind = AGX3_ISEL_UNIFORM,
               .false_kind = AGX3_ISEL_REGISTER,
               .sel_true = 10, .sel_false = 2, .wait_mask = 1u << 5},
    {0x02, 0x01, 0x0f, 0x86, 0x26, 0x0a, 0x17, 0xc2, 0x00, 0x02}, 10},
   {"u_false", {.dest = 0, .cmp0 = 0, .cmp1 = 7, .cmp1_immediate = true,
                .cond = AGX3_ISEL_ICOND_EQ, .explicit_hints = true,
                .discard = {true, false, false, false},
                .true_kind = AGX3_ISEL_REGISTER,
                .false_kind = AGX3_ISEL_UNIFORM,
                .sel_true = 2, .sel_false = 12, .wait_mask = 1u << 5},
    {0x02, 0x01, 0x0f, 0x86, 0x06, 0x02, 0x17, 0xc2, 0x20, 0x0c}, 10},
   {"u_both", {.dest = 0, .cmp0 = 0, .cmp1 = 7, .cmp1_immediate = true,
               .cond = AGX3_ISEL_ICOND_EQ, .explicit_hints = true,
               .discard = {true, false, false, false},
               .true_kind = AGX3_ISEL_UNIFORM,
               .false_kind = AGX3_ISEL_UNIFORM,
               .sel_true = 10, .sel_false = 12, .wait_mask = 1u << 5},
    {0x02, 0x01, 0x0f, 0x86, 0x26, 0x0a, 0x17, 0xc2, 0x20, 0x0c}, 10},
   {"u_true_imm", {.dest = 0, .cmp0 = 0, .cmp1 = 2,
                   .cond = AGX3_ISEL_ICOND_ULT, .explicit_hints = true,
                   .discard = {true, false, false, false},
                   .true_kind = AGX3_ISEL_UNIFORM,
                   .false_kind = AGX3_ISEL_IMMEDIATE,
                   .sel_true = 10, .sel_false = 3, .wait_mask = 1u << 5},
    {0x02, 0x01, 0x0f, 0x03, 0x22, 0x0a, 0x05, 0xc0, 0x20, 0x83}, 10},

   /* The boolean form every lowered ICMP becomes: `? 1u : 0u`, 16-bit dest. */
   {"bool1_0", {.dest = 2, .dest_16 = true, .cmp0 = 4, .cmp1 = 6,
                .cond = AGX3_ISEL_ICOND_ULT, .true_kind = AGX3_ISEL_IMMEDIATE, .false_kind = AGX3_ISEL_IMMEDIATE,
                .sel_true = 1, .sel_false = 0, .wait_mask = 1u << 5},
    {0x12, 0x05, 0x1d, 0x07, 0x22, 0x81, 0x05, 0xc0, 0x20, 0x80}, 10},
};

int
main(void)
{
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      const struct isel_vec *v = &vectors[i];
      uint8_t out[AGX3_ISEL_MAX_LENGTH];
      uint8_t len = 0;

      memset(out, 0xcd, sizeof(out));
      const bool ok =
         agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &v->d, out, &len);

      CHECK(ok);
      if (!ok)
         continue;
      CHECK(len == v->length);
      if (len != v->length)
         continue;

      ++checks;
      if (memcmp(out, v->bytes, len) != 0) {
         ++failures;
         fprintf(stderr, "FAIL %s: got", v->name);
         for (unsigned b = 0; b < len; ++b)
            fprintf(stderr, " %02x", out[b]);
         fprintf(stderr, ", want");
         for (unsigned b = 0; b < v->length; ++b)
            fprintf(stderr, " %02x", v->bytes[b]);
         fprintf(stderr, "\n");
      }
   }

   uint8_t out[AGX3_ISEL_MAX_LENGTH];
   uint8_t len = 0;
   struct agx3_isel_desc d = {.dest = 0, .cmp0 = 2, .cmp1 = 6,
                              .cond = AGX3_ISEL_ICOND_ULT,
                              .sel_true = 12, .sel_false = 16};

   /* Condition 0 is not a condition, and 3 appears in Apple's corpus with an
    * unknown meaning -- the control-flow core refuses it too. */
   d.cond = (enum agx3_isel_icond)0;
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.cond = (enum agx3_isel_icond)3;
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.cond = (enum agx3_isel_icond)7;
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.cond = AGX3_ISEL_ICOND_ULT;

   /* Misaligned or out-of-file 32-bit operands. */
   d.cmp0 = 3;
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.cmp0 = 2;
   d.sel_true = 256;
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.sel_true = 12;

   /* A 16-bit result takes single halves, so an odd index is fine there but
    * never for a 32-bit operand. */
   d.dest_16 = true;
   d.sel_true = 7;
   d.sel_false = 6;
   CHECK(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.dest_16 = false;
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.sel_true = 12;
   d.sel_false = 16;
   CHECK(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));

   /* Uniform shapes outside the witnesses are refused. */
   {
      struct agx3_isel_desc u = {.dest = 0, .cmp0 = 8, .cmp0_uniform = true,
                                 .cmp1 = 0, .cond = AGX3_ISEL_ICOND_ULT,
                                 .explicit_hints = true,
                                 .sel_true = 2, .sel_false = 4};
      CHECK(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &u, out, &len));
      u.cmp1_uniform = true; /* uniform/uniform comparands */
      CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &u, out, &len));
      u.cmp1_uniform = false;
      u.cmp1_immediate = true; /* uniform/immediate comparands */
      CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &u, out, &len));
      u.cmp1_immediate = false;
      u.true_kind = AGX3_ISEL_UNIFORM; /* uniform comparand and select */
      u.sel_true = 10;
      CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &u, out, &len));
      u.true_kind = AGX3_ISEL_REGISTER;
      u.sel_true = 2;
      u.cmp0 = 9; /* misaligned uniform word */
      CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &u, out, &len));
      u.cmp0 = 256; /* high uniform file */
      CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &u, out, &len));
      u.cmp0 = 8;
      u.dest_16 = true; /* no 16-bit uniform witness */
      CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &u, out, &len));
   }

   /* Immediates wider than a byte, and AGX3c, are not witnessed. */
   d.true_kind = AGX3_ISEL_IMMEDIATE;
   d.false_kind = AGX3_ISEL_IMMEDIATE;
   d.sel_true = 256;
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.sel_true = 1;
   d.sel_false = 0;
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, out, &len));
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX2, &d, out, &len));

   /* Waits: slot 1 is f98, slot 6 is f103, and the two differ. */
   d.dest_16 = false;
   d.true_kind = AGX3_ISEL_REGISTER;
   d.false_kind = AGX3_ISEL_REGISTER;
   d.sel_true = 12;
   d.sel_false = 16;
   d.cond = AGX3_ISEL_ICOND_ULT;
   uint8_t slot6[AGX3_ISEL_MAX_LENGTH], slot1[AGX3_ISEL_MAX_LENGTH];
   uint8_t l6 = 0, l1 = 0;
   d.wait_mask = 1u << 5;
   CHECK(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, slot6, &l6));
   d.wait_mask = 1u << 0;
   CHECK(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, slot1, &l1));
   CHECK(l6 == l1 && memcmp(slot6, slot1, l6) != 0);

   /* A slot-6 wait is exactly the witnessed `ult` instruction; with no wait
    * at all those bits clear. */
   uint8_t nowait[AGX3_ISEL_MAX_LENGTH];
   uint8_t ln = 0;
   d.wait_mask = 0;
   CHECK(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, nowait, &ln));
   CHECK(ln == l6 && memcmp(nowait, slot6, ln) != 0);

   d.wait_mask = 1u << 6;
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, nowait, &ln));
   d.wait_mask = 0;

   /* The compact native wait selector has single slots and paired slots1+2.
    * Other combinations require separate native retirement by the caller. */
   d.explicit_hints = true;
   d.wait_mask = 63;
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.wait_mask = 3;
   CHECK(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.cache[0] = d.discard[0] = true;
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.cache[0] = false;
   d.cmp1_immediate = true;
   d.cmp1 = 256;
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.cmp1 = 255;
   CHECK(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.cmp_16 = true;
   d.cmp0 = 255;
   CHECK(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.cmp0 = 256;
   CHECK(!agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   /* Fresh G17P bit_select and bit_negate witnesses. These establish the
    * mask-test condition and selected float sign bits independently of trig. */
   static const struct {
      struct agx3_bittest_desc d;
      uint8_t bytes[10];
   } bit_vectors[] = {
      {{.dest=0,.src={4,6,8},.mask=1,.cache={1,1,1},.wait_mask=32},
       {0x02,0x85,0x07,0x80,0x06,0x86,0x16,0xc2,0x00,0x88}},
      {{.dest=2,.src={4,6,8},.mask=2,.cache={1,1,1}},
       {0x12,0x85,0x07,0x82,0x06,0x86,0x06,0x02,0x00,0x88}},
      {{.dest=4,.src={4,6,8},.mask=4,.discard={1,1,1}},
       {0x22,0x05,0x0f,0x84,0x86,0x06,0x06,0x02,0x80,0x08}},
      {{.dest=0,.src={2,4,6},.mask=1,.negate={1,0},
        .cache={1,1,1},.wait_mask=32},
       {0x02,0x83,0x03,0x80,0x16,0x84,0x16,0xc2,0x00,0x86}},
      {{.dest=2,.src={2,4,6},.mask=2,.negate={0,1},.discard={1,1,1}},
       {0x12,0x03,0x0b,0x82,0x86,0x04,0x06,0x02,0x90,0x06}},
   };
   for (unsigned i = 0; i < sizeof(bit_vectors)/sizeof(bit_vectors[0]); ++i) {
      CHECK(agx3_bittest_encode(AGX_ISA_ENCODING_AGX3B, &bit_vectors[i].d,
                               out, &len));
      CHECK(len == 10 && memcmp(out, bit_vectors[i].bytes, 10) == 0);
   }
   struct agx3_bittest_desc bit = bit_vectors[0].d;
   CHECK(!agx3_bittest_encode(AGX_ISA_ENCODING_AGX3C, &bit, out, &len));
   CHECK(len == 0);
   bit.mask = 0;
   CHECK(!agx3_bittest_encode(AGX_ISA_ENCODING_AGX3B, &bit, out, &len));
   bit.mask = 1;
   bit.negate[0] = bit.negate[1] = true;
   CHECK(!agx3_bittest_encode(AGX_ISA_ENCODING_AGX3B, &bit, out, &len));
   bit = bit_vectors[0].d;
   bit.src[0] = 128;
   CHECK(!agx3_bittest_encode(AGX_ISA_ENCODING_AGX3B, &bit, out, &len));
   printf("%u checks, %u failures\n", checks, failures);
   return failures ? 1 : 0;
}
