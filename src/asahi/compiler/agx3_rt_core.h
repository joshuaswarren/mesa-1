/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 ray-traversal instruction core: composer-driven encoders for the five
 * hardware operations Apple's compiler emits for ray tracing, plus the
 * barrier that prefixes the BVH-update expansions.
 *
 *   intersect_ray   12B (AGX3b) / 14B (AGX3c)
 *   continue_ray     6B (AGX3b) /  8B (AGX3c)
 *   incStk           8B on both encodings
 *   barrier          6B on both encodings (the RT-expansion scope form)
 *   update_node      4B (AGX3b) /  8B (AGX3c)
 *   update_bvh       8B, AGX3c witness only -- AGX3b is refused
 *
 * Everything here is encoded through the generic table encoder
 * (agx3_compose, generated from Apple's own llvm::AGX3 encoder tables) from
 * per-form baselines recovered by decoding the witnessed corpus bytes, so a
 * wrong baseline cannot silently produce a valid-looking instruction.  The
 * witnesses are the same ones the byte-level codecs in agx3_pack.c pin:
 *
 *  - intersect_ray / continue_ray: decoded G16G/G17S Apple instructions plus
 *    the pinned five-kernel RT ladders that agx3_pack_{intersect,continue}_ray
 *    accept.
 *  - incStk, barrier, update_node, update_bvh: the exact fixed witnesses in
 *    agx3_pack.c (agx3_pack_inc_stk_fixed_witness, agx3_pack_update_node,
 *    agx3_pack_update_bvh_fixed_witness), decoded to composer fields and
 *    re-encoded byte-exactly.
 *
 * Fail-closed boundaries -- what is deliberately NOT encodable here, and
 * which witness is missing:
 *
 *  - update_bvh on AGX3b: Apple's tables spell the form, but no AGX3b
 *    update_bvh instruction exists in any pinned corpus.  Missing witness: a
 *    G15/G16/G17P native binary containing the llvm.agx3.update.bvh
 *    expansion.  Refused.
 *  - Operands on update_node / update_bvh / incStk: the intrinsics take
 *    arguments, but no corpus differential correlates any argument to a
 *    machine field (incStk's sole nonzero Apple-table operand decodes as
 *    f24=4 with no value ladder).  Missing witness: operand-varying
 *    differential pairs.  Only the canonical fixed bytes are encodable.
 *  - intersect_ray f77 is no longer anonymous: a concurrency differential
 *    over two to five named ray queries built for applegpu_g17p moves it with
 *    the ray query and with nothing else, so it is a per-query allocated
 *    resource id, reused once a traversal finishes rather than a dense index.
 *    agx3_rt_intersect_ray_query encodes it.
 *  - f5's "ray state" reading is REFUTED for intersect_ray.  It does not
 *    discriminate queries: it stays within {0, 4} with one, two or three
 *    queries live, and with four concurrent queries every intersect_ray
 *    still carries f5=0.  It is a 13-bit field per Apple's tables and every
 *    value 0..8191 encodes, but Apple emits only 0 and 4, one at each of the
 *    two intersect_ray sites a single traversal contains.  The name
 *    `state_register` on the older byte-level entry points predates this
 *    measurement.
 *  - intersect_ray f63 and f92, and the continue_ray context fields, are
 *    still held at their corpus values.  Missing witness: semantic
 *    differentials.  Ruled out for f92: it does not track the result field
 *    (six kernels each reading one committed field emit a byte-identical
 *    instruction despite differing in total code length), and it does not
 *    move with register pressure.  Recorded as refuted because it looks
 *    convincing: f92 = f78 - 4 holds across eight consecutive kernels and is
 *    FALSE -- the geometry-cull and opacity-cull variants give -2.  f63 and
 *    f92 shift when the ray-setup allocation shifts; nothing further is
 *    established.
 *  - f63, f77/f78 and f92 are REGISTER OPERANDS of
 *    intersect_ray, not payload or per-query descriptors, and f84/f85 are
 *    their operand-class selectors.  Two independent measurements against
 *    `metal-nt -arch applegpu_g17p`:
 *
 *      * `fieldbits` gives the layout directly -- f77 is 7 bits at 60..66
 *        under f84=2, f78 is 9 bits at the SAME start bit 60 under f84=3, and
 *        f84 accepts 1, 2 and 3.  Two widths of one field, which is the shape
 *        of a source-operand class selector.
 *      * A deliberate breaker, built because the per-query reading above
 *        survived several kernels: two kernels with a byte-identical query,
 *        one of them holding 24 extra floats live across the intersect.
 *        f63, f78 and f92 ALL shifted by +4 under that register pressure
 *        while every single (f70, f68, f71) SCS tuple stayed identical.
 *
 *    So the "per-query allocated resource id" reading of f77 above is an
 *    artefact: adding a query moves register allocation, and f77 follows the
 *    allocation.  An earlier six-point pressure ladder concluded the opposite
 *    because it held its extra values live around the wrong instruction.
 *    The entry points that expose f77 still encode witnessed bytes, so they
 *    remain correct as byte generators; only the NAME and the causal story
 *    are wrong, and a lowering must not treat f77 as a query identity.
 *    (The f92 = f78 - 4 refutation below stands and is now explained: both
 *    are register indices that usually, but not always, move together.)
 *  - Scoreboard/wait variation on any RT form: every witnessed RT
 *    instruction carries the same wait/hint bits, so no wait mask is
 *    exposed.  Missing witness: an RT instruction with a different wait set.
 *  - Scheduling, SCS layout, traversal lowering, the shared-memory and
 *    control-flow instructions between the RT ops, submission, and any API
 *    capability are outside this encoder, exactly as for agx3_pack.c.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Longest emission: AGX3c intersect_ray and the AGX3c barrier+update
 * expansions are all 14 bytes. */
#define AGX3_RT_CORE_MAX_LENGTH 14u

/*
 * Encode intersect_ray with the canonical anonymous-field context.
 * `state_register` is Apple's varying f5 field; accepted values are exactly
 * the corpus differential pairs: {0, 4} on AGX3b, {0, 8} on AGX3c.
 */
bool agx3_rt_intersect_ray(enum agx_isa_encoding enc, uint16_t state_register,
                           uint8_t out[AGX3_RT_CORE_MAX_LENGTH], uint8_t *len);

/*
 * Replay one witnessed intersect_ray corpus site, including the sites whose
 * anonymous scheduling-context fields differ from the canonical tuple
 * (AGX3b site 2 carries f92=130 instead of 134; AGX3c sites 2 and 3 carry
 * the f17/f28 variants).  Sites are indexed in corpus-vector order; sites 0
 * and 1 coincide with agx3_rt_intersect_ray at each encoding's two accepted
 * state registers.  No claim is made about what selects one context over
 * another -- that witness is missing, so the tuples are only enumerable.
 */
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
bool agx3_rt_intersect_ray_query(enum agx_isa_encoding enc, uint16_t site,
                                 uint16_t per_query, uint16_t context,
                                 uint8_t out[AGX3_RT_CORE_MAX_LENGTH],
                                 uint8_t *len);

#ifdef __cplusplus
}
#endif
