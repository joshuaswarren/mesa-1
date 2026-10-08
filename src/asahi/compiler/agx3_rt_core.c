/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 ray-traversal instruction core.  See agx3_rt_core.h for the grounding
 * contract.  Every baseline below was recovered by decoding the witnessed
 * corpus bytes against Apple's encoder tables and re-encoding byte-exactly;
 * the composer fails closed on any contradiction, and the test replays the
 * exact bytes for every accepted input.
 */

#include "agx3_rt_core.h"

#include "agx3_binder.h"

#include <string.h>

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}
#define NELEM(a) (sizeof(a) / sizeof((a)[0]))

static bool
enc_ok(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3B || e == AGX_ISA_ENCODING_AGX3C;
}

static bool
args_ok(const uint8_t *out, const uint8_t *len)
{
   return out != NULL && len != NULL;
}

static bool
in_set(uint16_t value, const uint16_t *set, unsigned n)
{
   for (unsigned i = 0; i < n; ++i) {
      if (set[i] == value)
         return true;
   }
   return false;
}

/*
 * Compose `mnemonic` from a fixed baseline plus overrides, zeroing the
 * caller's buffer on refusal so speculative callers stay transactional.
 */
static bool
rt_compose(enum agx_isa_encoding enc, const char *mnemonic,
           const struct agx3_field_kv *baseline, unsigned nbaseline,
           const struct agx3_field_kv *overrides, unsigned noverrides,
           uint8_t out[AGX3_RT_CORE_MAX_LENGTH], uint8_t *len)
{
   if (!args_ok(out, len))
      return false;
   memset(out, 0, AGX3_RT_CORE_MAX_LENGTH);
   *len = 0;

   if (!enc_ok(enc))
      return false;

   if (!agx3_bind_compose(enc, mnemonic, 0, baseline, nbaseline, overrides,
                          noverrides, out, len)) {
      memset(out, 0, AGX3_RT_CORE_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

/*
 * intersect_ray canonical anonymous-field context, from corpus vectors 561
 * (AGX3b, G16G) and 1351 (AGX3c, G17S).  f5 is the overridden operand.
 */
static const struct agx3_field_kv b_intersect_base[] = {
   KV(9, 5),   KV(12, 6),  KV(13, 1),  KV(30, 1),  KV(49, 32), KV(55, 3),
   KV(56, 3),  KV(63, 100), KV(69, 3), KV(70, 2),  KV(78, 116), KV(84, 3),
   KV(85, 2),  KV(92, 134), KV(98, 3), KV(99, 1),
};

static const struct agx3_field_kv c_intersect_base[] = {
   KV(9, 5),   KV(10, 2),  KV(12, 8),  KV(13, 1),  KV(14, 2),  KV(17, 46),
   KV(24, 2),  KV(25, 3),  KV(29, 92), KV(30, 1),  KV(49, 32), KV(55, 3),
   KV(56, 3),  KV(63, 100), KV(69, 3), KV(70, 2),  KV(85, 1),  KV(99, 1),
};

/*
 * Encode intersect_ray with the canonical anonymous-field context.
 * `state_register` is Apple's varying f5 field; accepted values are exactly
 * the corpus differential pairs: {0, 4} on AGX3b, {0, 8} on AGX3c.
 */
static bool
agx3_rt_intersect_ray(enum agx_isa_encoding enc, uint16_t state_register,
                      uint8_t out[AGX3_RT_CORE_MAX_LENGTH], uint8_t *len)
{
   /* The corpus differential pairs vary only f5, with these exact values. */
   static const uint16_t b_regs[] = {0, 4};
   static const uint16_t c_regs[] = {0, 8};

   if (!args_ok(out, len))
      return false;

   const bool is_b = enc == AGX_ISA_ENCODING_AGX3B;
   if (!enc_ok(enc) ||
       !in_set(state_register, is_b ? b_regs : c_regs, 2)) {
      memset(out, 0, AGX3_RT_CORE_MAX_LENGTH);
      *len = 0;
      return false;
   }

   const struct agx3_field_kv ov[] = {KV(5, state_register)};
   return rt_compose(enc, "intersect_ray",
                     is_b ? b_intersect_base : c_intersect_base,
                     is_b ? NELEM(b_intersect_base) : NELEM(c_intersect_base),
                     ov, NELEM(ov), out, len);
}

/*
 * Every distinct witnessed intersect_ray site, in corpus-vector order.  The
 * overrides land on top of the canonical baseline above.
 *
 *   AGX3b: 561 (f5=0), 470/651 (f5=4), 263 (f5=0, f92=130)
 *   AGX3c: 1351 (f5=0), 1252 (f5=8), 1006 (f5=0, f17=4, f28=1),
 *          1453 (f5=8, f28=1)
 */
struct rt_site {
   struct agx3_field_kv ov[3];
   uint8_t nov;
};

static const struct rt_site b_intersect_sites[] = {
   {{KV(5, 0)}, 1},
   {{KV(5, 4)}, 1},
   {{KV(5, 0), KV(92, 130)}, 2},
};

static const struct rt_site c_intersect_sites[] = {
   {{KV(5, 0)}, 1},
   {{KV(5, 8)}, 1},
   {{KV(5, 0), KV(17, 4), KV(28, 1)}, 3},
   {{KV(5, 8), KV(28, 1)}, 2},
};

/*
 * The invariant context shared by every witnessed AGX3b ray-query
 * intersect_ray; f5, f77 and f92 are the three fields that vary.
 */
static const struct agx3_field_kv b_intersect_query_base[] = {
   KV(9, 5),   KV(12, 6),  KV(13, 1),  KV(30, 1),  KV(49, 32),
   KV(55, 3),  KV(56, 3),  KV(63, 100), KV(69, 3), KV(70, 2),
   KV(84, 2),  KV(85, 2),  KV(88, 1),  KV(98, 3),  KV(99, 1),
};

/*
 * Encode one AGX3b inline-ray-query intersect_ray from the three fields that
 * actually vary across Apple's ray-query lowering.
 *
 * `per_query` is Apple's f77.  It tracks the ray query rather than anything
 * else: two and three separately named queries give two and three distinct
 * values, while the SCS ray staging offsets stay 0 and 16 for every query and
 * a six-point register-pressure ladder leaves the whole encoding
 * byte-identical, so it is neither the ray's address nor an allocator-assigned
 * register.
 *
 * It is NOT a dense per-query index, and an earlier revision of this comment
 * overstated it as one.  Extending the sweep to four and five named queries
 * gives only three and four distinct values: these kernels traverse their
 * queries one after another, so the compiler reuses a value once a traversal
 * has finished.  Read it as a per-query allocated resource id whose lifetime
 * is the traversal, not as a query number.  Written as an array traversed in a
 * loop the queries share one code path and this whole field is absent, the
 * encoding taking a different form.
 *
 * Its encodable range is exactly 0..191, the same bound agx3_scs_core applies
 * to an AGX3b scalar register index, which is consistent with a half-register
 * index but is not proof of one, so no register name is given here.
 *
 * `site` is Apple's f5, a full 13-bit field (0..8191 all encode).  It is NOT
 * the per-query discriminator: it stays within {0, 4} whether one, two or
 * three queries are live, taking one value at each of the two intersect_ray
 * sites a single query's traversal contains.
 *
 * `context` is Apple's f92.  It varies between kernels (124 in the two-query
 * corpus, 122 in the three-query one) and no differential explains it, so it
 * is carried verbatim and deliberately unnamed.
 *
 * f77 and f78 are the SAME PHYSICAL BITS, and f84 chooses which identity
 * applies.  With f84=2 (and f88 set) the composer takes f77 and refuses f78;
 * with f84=3 (and f88 clear) it takes f78 and refuses f77; it never accepts
 * both.  Both deposit identically -- byte 6 high nibble and byte 7 -- and they
 * differ only in range: f77 stops at 191, while every f78 value tested up to
 * 256 encodes.
 *
 * Which form Apple emits is NOT simply a function of how many queries are
 * live.  A single-query kernel against a primitive acceleration structure
 * uses the f84=3/f78 form and two or more queries use the f84=2/f77 form,
 * but a SINGLE-query kernel against an instance acceleration structure also
 * uses f77.  Instancing selects it too, so what f84 keys on is unmapped.
 * The per-query differential above is expressed in f77 because that is the
 * form those kernels emit.
 *
 * This entry point emits the f84=2/f77 variant, the one witnessed whenever
 * more than one query is live.  The single-query f78 variant is reachable
 * through the enumerated witnessed-site replay, not here.
 *
 * Values outside the composer's tables are refused by the composer itself.
 * AGX3c is refused: this differential is an AGX3b/G17P corpus.
 */
static bool
agx3_rt_intersect_ray_query(enum agx_isa_encoding enc, uint16_t site,
                            uint16_t per_query, uint16_t context,
                            uint8_t out[AGX3_RT_CORE_MAX_LENGTH], uint8_t *len)
{
   if (!args_ok(out, len))
      return false;

   /* f5 is 13 bits; f77 is refused at 192 and above. */
   if (enc != AGX_ISA_ENCODING_AGX3B || site > 8191u || per_query > 191u) {
      memset(out, 0, AGX3_RT_CORE_MAX_LENGTH);
      *len = 0;
      return false;
   }

   const struct agx3_field_kv ov[] = {
      KV(5, site),
      KV(77, per_query),
      KV(92, context),
   };
   return rt_compose(enc, "intersect_ray", b_intersect_query_base,
                     NELEM(b_intersect_query_base), ov, NELEM(ov), out, len);
}

/*
 * continue_ray canonical context, from corpus vectors 275 (AGX3b) and 1017
 * (AGX3c).  f5 is the overridden operand.
 */
static const struct agx3_field_kv b_continue_base[] = {
   KV(9, 5), KV(12, 6), KV(13, 1), KV(16, 1),
};

static const struct agx3_field_kv c_continue_base[] = {
   KV(9, 5), KV(10, 2), KV(12, 8), KV(13, 1), KV(14, 2), KV(15, 34),
   KV(16, 1),
};

