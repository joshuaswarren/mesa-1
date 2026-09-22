/* SPDX-License-Identifier: MIT */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "asahi/lib/agx_chip.h"

/* Native same-width integer arithmetic with a uniform operand. */
/* Matches the AGX3B_NUM_REGS allocation limit in agx_compiler.h. Encodable
 * eight-bit register indices above this limit are not allocatable GPRs. */
#define AGX3_UNIFORM_INTEGER_GPRS 192
enum agx3_uniform_integer_op {
   AGX3_UI_ADD,
   AGX3_UI_SUB,
   AGX3_UI_MAD,
   AGX3_UI_AND,
   AGX3_UI_OR,
   AGX3_UI_XOR,
   AGX3_UI_NAND,
   AGX3_UI_NOR,
   AGX3_UI_ANDC,
};
enum agx3_uniform_integer_kind {
   AGX3_UI_ZERO = 1,
   AGX3_UI_REGISTER = 2,
   AGX3_UI_UNIFORM = 3,
};
struct agx3_uniform_integer_src {
   enum agx3_uniform_integer_kind kind;
   uint16_t value;
   bool cache, discard;
};
struct agx3_uniform_integer_desc {
   enum agx3_uniform_integer_op op;
   bool half, dest_cache;
   uint16_t dest;
   /* Ordinary expression order. MAD has three sources; binary ops have two.
    * ANDC means src[0] & ~src[1] and only U/R is currently witnessed. */
   struct agx3_uniform_integer_src src[3];
   /* ADD computes src0 + (src1 << shift). Only shifts zero and three are
    * witnessed here; a shifted add is never commuted. */
   uint8_t shift, wait_mask;
};
/* Unsupported domains leave output and length unchanged. */
#ifdef __cplusplus
extern "C" {
#endif
bool agx3_uniform_integer_encode(enum agx_isa_encoding encoding,
                                 const struct agx3_uniform_integer_desc *d,
                                 uint8_t output[16], uint8_t *length);
#ifdef __cplusplus
}
#endif
