/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 coverage / depth-stencil writeout encoders: `movmsk` and `depthf`.
 *
 * These are the AGX3 counterparts of AGX2's `sample_mask` and `zs_emit`. Both
 * are six bytes on AGX3b and eight on AGX3c, and both have the same shape: one
 * register operand naming the staged writeout word, a kill bit on it, the
 * scoreboard slot the instruction takes, and the per-slot wait mask -- plus,
 * for `depthf`, the depth/stencil enable pair.
 *
 * AGX2 spells both instructions with *two* operands (sample_mask's target and
 * live masks; zs_emit's sample mask and z/s value). AGX3 carries a single
 * register, and every Apple witness stages it with an asynchronous `copy` or
 * `ldimm` immediately before. What that register must contain is therefore a
 * separate contract from the encoding, and only part of it is grounded; see
 * agx3_pack_binary.c for exactly which IR shapes are routed here and which
 * stay refused.
 *
 * Operand roles, and the evidence for each:
 *
 *  - f9 = 5 selects the 13-bit operand field f5. Baseline in every witness of
 *    both mnemonics on both encodings.
 *  - f5 = the staged word's half-register index. Established by def-use, not
 *    by correlation: the corpus `discard_msaa4` shader contains two `movmsk`
 *    instructions, at f5=0 (defined by `ldimm r0` immediately before) and
 *    f5=2 (defined by `copy r1 <- r2`). The same 13-bit field with the same
 *    f9=5 selector is a half-register operand in the independently grounded
 *    `uvsw` and `copy` forms (agx3_vary_core, agx3_tex_core).
 *  - f13 = kill (last use) on that operand, as on uvsw's value. It is 1 in
 *    every 1x witness and 0 in the 4x depth witness, matching a liveness bit
 *    rather than a constant.
 *  - f12 = the scoreboard slot, through the encoder LUT the composer applies.
 *  - f17..f22 (AGX3b) / f17..f24 (AGX3c) = the per-slot wait mask, one bit per
 *    slot starting at slot 1, exactly as agx3_pixel_core spells it for pixrel.
 *  - depthf f25 / f26 = depth written / stencil written; AGX3c additionally
 *    carries the redundant mode selector f35 (1 depth, 2 stencil, 3 both).
 *
 * Every field that no role names is left at the baseline the composer's own
 * strict check requires, so a wrong value cannot silently produce a
 * valid-looking instruction.
 *
 * Evidence: research/gpu/GPU_AGX3_TILE_OPS_2026-08-24.md section 4, the
 * tileops corpus at research/gpu/isa-re/gen-x/tileops/corpus/tile{1,2}, the
 * smp-frag-re z/s probes, and the fresh discard / sample-mask probe ladder
 * documented in /tmp/g17p-re/notes/mesa-agx3-fs-discard-depthwrite.md.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* movmsk and depthf are 6 bytes on AGX3b and 8 on AGX3c. */
#define AGX3_ZS_MAX_LENGTH 8u

/*
 * The 32-bit word Apple's compiler stages for an unconditional kill of every
 * sample -- `discard_fragment()` with no surviving coverage, which is what
 * Mesa's `sample_mask(target = ALL_SAMPLES, live = 0)` means.
 *
 * Directly witnessed, not derived: three independent probes (an unconditional
 * `discard_fragment()`, the same with a void fragment function, and
 * `[[sample_mask]] = 0`) all compile to `ldimm r0, 0xFFFFFF00` immediately
 * followed by `movmsk r0`, on g15g/g16g/g17s alike, and the corpus
 * `discard_msaa4` shader stages the same constant for its first movmsk. No
 * other movmsk operand value is witnessed as a constant.
 */
#define AGX3_ZS_KILL_ALL_SAMPLES 0xFFFFFF00u

struct agx3_zs_desc {
   /* Half-register index of the staged writeout word. 32-bit, so even. */
   uint16_t value;
   /* The staged word is dead after this instruction. */
   bool value_kill;
   /* Scoreboard slot this instruction takes: 1..6 (AGX3b) / 1..8 (AGX3c). */
   uint8_t slot;
   /* Bit i waits scoreboard slot i+1. */
   uint16_t wait_mask;
};

/* `movmsk`: write the staged coverage word. */
bool agx3_zs_encode_sample_mask(enum agx_isa_encoding enc,
                                const struct agx3_zs_desc *d,
                                uint8_t out[AGX3_ZS_MAX_LENGTH], uint8_t *len);

/*
 * `depthf`: trigger depth/stencil writeout at the staged payload base. G17P
 * depth-only payloads contain coverage32 at value and depth32 at value + 2.
 * The caller must publish both words before this instruction. At least
 * one of `depth`/`stencil` must be set; all three combinations are witnessed.
 */
bool agx3_zs_encode_depth_stencil(enum agx_isa_encoding enc,
                                  const struct agx3_zs_desc *d, bool depth,
                                  bool stencil,
                                  uint8_t out[AGX3_ZS_MAX_LENGTH],
                                  uint8_t *len);

#ifdef __cplusplus
}
#endif
