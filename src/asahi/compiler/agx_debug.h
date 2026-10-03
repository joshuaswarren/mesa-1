/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>

#include "util/macros.h"

#ifdef __cplusplus
extern "C" {
#endif

/* clang-format off */
enum agx_compiler_dbg {
   /* bit 0 unused */
   AGX_DBG_SHADERS     = BITFIELD_BIT(1),
   AGX_DBG_SHADERDB    = BITFIELD_BIT(2),
   AGX_DBG_VERBOSE     = BITFIELD_BIT(3),
   AGX_DBG_INTERNAL    = BITFIELD_BIT(4),
   AGX_DBG_NOVALIDATE  = BITFIELD_BIT(5),
   AGX_DBG_NOOPT       = BITFIELD_BIT(6),
   AGX_DBG_WAIT        = BITFIELD_BIT(7),
   AGX_DBG_NOPREAMBLE  = BITFIELD_BIT(8),
   AGX_DBG_DEMAND      = BITFIELD_BIT(9),
   AGX_DBG_NOSCHED     = BITFIELD_BIT(10),
   AGX_DBG_SPILL       = BITFIELD_BIT(11),
   AGX_DBG_NOPROMOTE   = BITFIELD_BIT(12),
   AGX_DBG_REFUSE      = BITFIELD_BIT(13),
   AGX_DBG_NATIVE_UNIFORMS = BITFIELD_BIT(15),
};
/* clang-format on */

uint64_t agx_get_compiler_debug(void);

/* v56d: why the last agx_compile_shader_nir() on THIS thread returned false
 * ("" when it succeeded), for the driver's "cannot encode" error. */
const char *agx_last_compile_refusal(void);

/* v56c: the G17P fragment constant program (the AGX_DBG_NATIVE_UNIFORMS
 * behaviour, fragment shaders only) is the default. AGX_G17P_FS_CPROG=0, or
 * the master AGX_G17P_V56C_DEFAULTS=0, restores v56b; AGX_MESA_DEBUG=
 * native_uniforms still forces the all-stage debug behaviour. */
bool agx_g17p_fs_cprog_default(void);

/* v57 (neo-mc-shaders-mesa-v57) codegen flags, opt-in. The driver's codegen
 * environment hash keeps their cached binaries apart. */
enum agx_v57_flag {
   /* SMP reads its texture/sampler index operands (the indexed-heap indices a
    * constant program leaves in uniforms) straight from the uniform file:
    * Apple's G17P witness is the MoltenVK argument-buffer fragment
    * (fields 103:3/132:3, 97:u12, 126:u16, widths 104:2/133:2). */
   AGX_V57_TEX_UNIFORM_OPERANDS = 1u << 0,
};
uint32_t agx_get_v57_flags(void);
bool agx_g17p_fs_native_uniforms(void);

/* G17P native-encoding code generation choices. Unlike the debug flags above
 * they change emitted binaries without disabling caching, so every shader
 * cache key must include agx_get_codegen_flags(). Each is read once from its
 * environment variable and defaults on; set it to false to disable.
 */
enum agx_codegen_flag {
   AGX_CODEGEN_G17P_IMMEDIATE_SHIFTS          = BITFIELD_BIT(0),
   AGX_CODEGEN_G17P_PROMOTE_INTEGER_CONSTANTS = BITFIELD_BIT(1),
   AGX_CODEGEN_G17P_COMPACT_BITOPS            = BITFIELD_BIT(2),
   AGX_CODEGEN_G17P_SHIFTED_ADD               = BITFIELD_BIT(3),
   AGX_CODEGEN_G17P_UNIFORM_CF_HINTS          = BITFIELD_BIT(4),
   AGX_CODEGEN_G17P_PIXEL_COORD_FOLD          = BITFIELD_BIT(5),
   AGX_CODEGEN_G17P_INT_DEST_CACHE            = BITFIELD_BIT(6),
   /* v46 execution-efficiency candidates, all default off (opt-in). */
   AGX_CODEGEN_G17P_TIB_UNPACK_DIRECT         = BITFIELD_BIT(7),
   AGX_CODEGEN_G17P_BITOP_IMMEDIATES          = BITFIELD_BIT(8),
   AGX_CODEGEN_G17P_EXTRACT16_BFE             = BITFIELD_BIT(9),
   AGX_CODEGEN_G17P_ILP_SCHEDULE              = BITFIELD_BIT(10),
   /* v49 light-fragment candidates (neo-gpu-exec-mesa-v49), default off. */
   AGX_CODEGEN_G17P_SR32_PIXEL_COORD          = BITFIELD_BIT(11),
   AGX_CODEGEN_G17P_BFE_IMMEDIATE             = BITFIELD_BIT(12),
   AGX_CODEGEN_G17P_UNIT_RANGE_BLEND          = BITFIELD_BIT(13),
   /* v52a ALU-issue candidates (neo-alu-mesa-v52a), default off. */
   AGX_CODEGEN_G17P_LOOP_ROTATE               = BITFIELD_BIT(14),
   AGX_CODEGEN_G17P_LOOP_HEAD_HINT            = BITFIELD_BIT(15),
   /* v53-c3: loop-exit compare placement (exit-compare cone at a fixed
    * distance before its CND) plus a register-cache hint on its counter. */
   AGX_CODEGEN_G17P_EXIT_COMPARE              = BITFIELD_BIT(16),
   /* msaa163: G18G four-sample colour rendering resolved at end of tile
    * (native run112 workload 8). AGX_G18_MSAA4, default off. Off, every G18G
    * multisampled draw keeps its existing refusal and no emitted byte
    * changes; on, only four-sample G18G tile programs and draws change. */
   AGX_CODEGEN_G18_MSAA4                      = BITFIELD_BIT(26),
   AGX_CODEGEN_G17P_COLOR_ILP                 = BITFIELD_BIT(17),
   /* frg-codegen (2026-09-30): native light-fragment tail and loop-entry
    * forms. Bit 18 was used by the unpromoted tile-wait experiment; it is
    * skipped so no cache identity is shared with that library. */
   AGX_CODEGEN_G17P_BANK_COALESCE             = BITFIELD_BIT(19),
   AGX_CODEGEN_G17P_UNIFORM_IMM_COMPARE       = BITFIELD_BIT(20),
   AGX_CODEGEN_G17P_TILE_WAIT_SINK            = BITFIELD_BIT(21),
   AGX_CODEGEN_G17P_FLTNORM_LIVENESS          = BITFIELD_BIT(22),
   AGX_CODEGEN_G17P_ENTRY_COMPARE             = BITFIELD_BIT(23),
   /* frg-codegen v2 (2026-10-01). */
   AGX_CODEGEN_G17P_EMPTY_THEN                = BITFIELD_BIT(24),
   /* frg-codegen v3 (2026-10-01): one-sample discard coverage word. */
   AGX_CODEGEN_G17P_DISCARD_WORD              = BITFIELD_BIT(25),
};

uint32_t agx_get_codegen_flags(void);

/* AGX_G18_MSAA4 (default false): see AGX_CODEGEN_G18_MSAA4. */
bool agx_g18_msaa4_enabled(void);

/* v50 agx3 footgun-audit candidates (neo-footgun-mesa-v50). A separate
 * namespace from agx_codegen_flag so it composes with later codegen bits;
 * Honeykrisp keys these into its shader cache at bits 35..39. All default
 * off until qualified on hardware, except the internal-shader scaled guard. */
enum agx_footgun_flag {
   /* hk: branch around the scaled-view texel conversion (kind == 0 skips). */
   AGX_FOOTGUN_G17P_SCALED_SAMPLE_GUARD = BITFIELD_BIT(0),
   /* compiler: nir_opt_uniform_atomics on AGX3B (SR34/SR36 grounded). */
   AGX_FOOTGUN_G17P_UNIFORM_ATOMICS     = BITFIELD_BIT(1),
   /* compiler: one centre RHW/1/W per fragment shader instead of one per
    * interpolated-input load. */
   AGX_FOOTGUN_G17P_SHARED_RHW          = BITFIELD_BIT(2),
   /* hk+compiler (v51a): hoist constant-index descriptor/push loads of the
    * fragment shader into one launch-time uniform DMA range. */
   AGX_FOOTGUN_G17P_UNIFORM_DMA         = BITFIELD_BIT(3),
   /* hk: the scaled-view guard for driver-internal (vk_meta) shaders only.
    * Default on; HK_G17P_INTERNAL_SCALED_GUARD=0 is its A/B off-switch. */
   AGX_FOOTGUN_G17P_INTERNAL_SCALED_GUARD = BITFIELD_BIT(4),
};

#define AGX_FOOTGUN_FLAG_BITS 5

/* v54 note: AGX_G17P_FS_STRUCTURE=1 now bundles only the five flags the lab
 * measured effective (TIB_UNPACK_DIRECT, PIXEL_COORD_FOLD, SR32_PIXEL_COORD,
 * UNIT_RANGE_BLEND, UNIFORM_DMA). The four v52-only flags below measured
 * null on light and stay individual diagnostics.
 *
 * v52 fragment-structure candidates (neo-fs-structure-mesa-v52), default
 * off. Honeykrisp keys them into its shader cache at bits 33, 59, 62, 63 in
 * the v52 tree and at bits 28..31 in v53-pre (v52b owns 33/59/62/63).
 *
 * AGX_G17P_FS_STRUCTURE=1 is a bundle: it turns on every flag below plus the
 * existing structure flags AGX_G17P_TIB_UNPACK_DIRECT,
 * AGX_G17P_PIXEL_COORD_FOLD, AGX_G17P_SR32_PIXEL_COORD,
 * AGX_G17P_UNIT_RANGE_BLEND and AGX_G17P_UNIFORM_DMA. The ALU/loop flags
 * (bit-op/BFE immediates, ILP) are left to their own switches. */
enum agx_fs_structure_flag {
   /* The unit-range blend proof accepts exact constants up to 1.0. */
   AGX_FS_STRUCTURE_UNIT_RANGE_CONST = BITFIELD_BIT(0),
   /* The fragment main part opens the colour-tile protocol (pixwait 512,1)
    * at its entry, where Apple's compiler places it; the linker drops the
    * epilog's copy, or the main's when no colour epilog follows. */
   AGX_FS_STRUCTURE_EARLY_COLOUR_GATE = BITFIELD_BIT(1),
   /* One-sample sample_mask/discard: fold a compile-time target (including
    * `x | c` with bit 0 of c set) and keep value instead of building them with
    * ALU ops and an IMAD select at run time. */
   AGX_FS_STRUCTURE_DISCARD_FOLD = BITFIELD_BIT(2),
   /* One fragment-input entry marker per linked fragment, on its first
    * input. The first pixel-coordinate read (SR 23/24, 32-bit) carries it,
    * as Apple's throughput fragment's first rld does (1ca1110601000000), and
    * the linker clears the epilog's LLDCOL first-load marker when an earlier
    * part already marked an input: Apple's throughput blend read
    * (67065404000001ce02000000) is unmarked after its marked rld, while its
    * RGB565 blend with no earlier input keeps the marker (670e5400...). */
   AGX_FS_STRUCTURE_INPUT_MARK = BITFIELD_BIT(3),
   /* v54 (neo-fs-blend-mesa-v54), not in the v52 bundle: compile the colour
    * epilog into the fragment shader at link time, as Apple's compiler
    * blends inline. Honeykrisp keeps the main shader's NIR and backend key;
    * at fast-link it appends the draw's epilog, turns every export /
    * load_exported pair into SSA values and compiles one fragment program.
    * It falls back to the split main + epilog whenever the monolithic
    * compile would change any uniform, rodata, varying or preamble layout
    * the draw path takes from the API shader, or fails to compile. */
   AGX_FS_STRUCTURE_INLINE_BLEND = BITFIELD_BIT(4),
   /* v54, with INLINE_BLEND: when the fragment's coverage update kills every
    * sample unconditionally, drop the unwritten-target preserve (tile
    * reload + restore). The preserve exists because a *covered* fragment
    * with no tile store dirties the tile; a fully killed fragment is not
    * covered. Hardware-unverified: the oracle must see the background. */
   AGX_FS_STRUCTURE_DISCARD_NO_RESTORE = BITFIELD_BIT(5),
};

#define AGX_FS_STRUCTURE_FLAG_BITS 6

uint32_t agx_get_fs_structure_flags(void);

uint32_t agx_get_footgun_flags(void);

/* v52b ALU-issue candidates (neo-alu-mesa-v52b). A separate namespace so the
 * existing cache-key layout is untouched; Honeykrisp keys these at bits 33,
 * 59, 62 and 63. All default off until qualified on hardware. */
enum agx_alu_flag {
   /* compiler: a loop/if integer compare reads a 32-bit uniform operand
    * directly (Apple icmp f18/f24=3) instead of copying it to a GPR. */
   AGX_ALU_G17P_UNIFORM_COMPARE = BITFIELD_BIT(0),
   /* packer: place a block's terminal loop compare right after the last
    * definition of its sources, leaving only the CND at the block end. */
   AGX_ALU_G17P_EARLY_COMPARE   = BITFIELD_BIT(1),
   /* compiler: register-cache discard hints from liveness over the physical
    * (exec-masked) CFG, so divergent blocks get safe last-use hints. */
   AGX_ALU_G17P_PHYSICAL_DISCARD = BITFIELD_BIT(2),
   /* RA: scalar temporaries rotate through free registers below the current
    * high-water mark instead of always taking the lowest free register. */
   AGX_ALU_G17P_SPREAD_TEMPS    = BITFIELD_BIT(3),
};

#define AGX_ALU_FLAG_BITS 4

uint32_t agx_get_alu_flags(void);

#ifdef __cplusplus
} /* extern C */
#endif
