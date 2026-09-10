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
 *  - intersect_ray / continue_ray: the decoded G16G/G17S corpus vectors in
 *    test/agx3_compose_corpus_vectors.c (intersect cases 263/470/561/651 and
 *    1006/1252/1351/1453; continue cases 275/482/573/663 and
 *    1017/1263/1361/1463) plus the pinned five-kernel RT corpus ladders that
 *    agx3_pack_{intersect,continue}_ray accept.  The test replays every
 *    accepted value byte-for-byte against those codecs' deposit scheme.
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
 *  - intersect_ray / continue_ray fields other than f5: the anonymous
 *    selector/hint fields are held at their corpus values (the alternate
 *    witnessed intersect_ray tuples are reachable only through the
 *    enumerated witnessed-site replay below).  Missing witness: semantic
 *    differentials for those fields.  f5 itself is a 13-bit half-register
 *    field per Apple's tables; its ray-state/SCS role is an inference from
 *    the surrounding lowering.
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
unsigned agx3_rt_intersect_ray_num_witnessed_sites(enum agx_isa_encoding enc);
bool agx3_rt_intersect_ray_witnessed_site(enum agx_isa_encoding enc,
                                          unsigned site,
                                          uint8_t out[AGX3_RT_CORE_MAX_LENGTH],
                                          uint8_t *len);

/*
 * Encode continue_ray.  `state_register` is the same Apple f5 field;
 * accepted values are the pinned five-kernel RT corpus ladders, identical to
 * agx3_pack_continue_ray: {0, 2, 4, 6, 8, 10} on AGX3b and
 * {0, 2, 8, 10, 12, 16, 18} on AGX3c.  Values Apple's tables can merely
 * spell (the composer would take any 13-bit value) are refused.
 */
bool agx3_rt_continue_ray(enum agx_isa_encoding enc, uint16_t state_register,
                          uint8_t out[AGX3_RT_CORE_MAX_LENGTH], uint8_t *len);

/*
 * Encode the fixed incStk instruction witnessed once per pinned
 * rt_rayquery_committed binary (composer fields AGX3b {f2=36, f9=2, f10=2,
 * f24=4} / AGX3c {f2=18, f9=2, f10=2, f14=3, f15=90, f24=4}, recovered from
 * the exact witness bytes).  No operand is exposed; see the header comment.
 */
bool agx3_rt_inc_stk(enum agx_isa_encoding enc,
                     uint8_t out[AGX3_RT_CORE_MAX_LENGTH], uint8_t *len);

/*
 * Encode incStk with its two varying operand fields.  AGX3b only.
 *
 * The header comment above records that incStk's operand codec was blocked on
 * a missing witness: "no corpus differential correlates any argument to a
 * machine field".  That witness now exists.  Compiling five ray-query kernels
 * and a 1/2/3/4-concurrent-query family for `applegpu_g17p` with Apple's own
 * `metal-nt` yields 24 distinct incStk encodings covering 24 distinct
 * (f2, f24) pairs, and f2/f24 vary independently: (46,5) and (46,8) share f2,
 * (62,352) and (62,520) share f2 with differing f24.  Every one of the 24 is
 * reproduced exactly by this encoder, and nothing outside f2/f24 varies across
 * them (the `07 00 54` prefix, the zero tail, f9=2 and f10=2 are invariant).
 *
 * What is witnessed is the DEPOSIT, not the meaning.  f2 is an 8-bit field
 * deposited in half-register units (byte 3 = f2/2), the same convention the
 * memory forms use for a register operand; f24 is a 13-bit field at bit 37.
 * Neither field's ray-traversal role is established -- deliberately no
 * `stack_depth`/`slot` name is invented for them here, because f5 on
 * intersect_ray was named `state_register` on an inference that the measured
 * G17P corpus does not support (see agx3_rt_intersect_ray).
 *
 * AGX3c is refused: the operand differential above is an AGX3b/G17P corpus and
 * no AGX3c incStk operand pair has been witnessed.
 */
bool agx3_rt_inc_stk_operands(enum agx_isa_encoding enc, uint16_t operand_f2,
                              uint16_t operand_f24,
                              uint8_t out[AGX3_RT_CORE_MAX_LENGTH],
                              uint8_t *len);

/*
 * Encode the exact barrier instruction Apple's compiler emits ahead of the
 * update_node / update_bvh operations (the RT-expansion scope form of the
 * general barrier).  Other barrier scopes belong to a future barrier core,
 * not here.
 */
bool agx3_rt_update_barrier(enum agx_isa_encoding enc,
                            uint8_t out[AGX3_RT_CORE_MAX_LENGTH], uint8_t *len);

/*
 * Encode the fixed update_node operation (the instruction alone, without the
 * preceding barrier).
 */
bool agx3_rt_update_node(enum agx_isa_encoding enc,
                         uint8_t out[AGX3_RT_CORE_MAX_LENGTH], uint8_t *len);

/*
 * Encode the fixed update_bvh operation (instruction alone).  AGX3c only;
 * AGX3b has no witness and is refused.
 */
bool agx3_rt_update_bvh(enum agx_isa_encoding enc,
                        uint8_t out[AGX3_RT_CORE_MAX_LENGTH], uint8_t *len);

/*
 * The complete witnessed expansions: barrier followed by the update
 * operation.  Byte-identical to agx3_pack_update_node and
 * agx3_pack_update_bvh_fixed_witness respectively (asserted by the test),
 * but built from the composer rather than stored bytes.
 */
bool agx3_rt_update_node_expansion(enum agx_isa_encoding enc,
                                   uint8_t out[AGX3_RT_CORE_MAX_LENGTH],
                                   uint8_t *len);
bool agx3_rt_update_bvh_expansion(enum agx_isa_encoding enc,
                                  uint8_t out[AGX3_RT_CORE_MAX_LENGTH],
                                  uint8_t *len);

#ifdef __cplusplus
}
#endif
