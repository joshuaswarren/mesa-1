/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/*
 * Apple GPU identity, instruction-set generation and binary encoding.
 *
 * These are three different questions and the driver has historically only
 * asked the first one, because every part it supported answered the other two
 * the same way. That stops being true at M3:
 *
 *   - G13 (M1) and G14 (M2) are the AGX2 instruction set. That is what
 *     agx_pack.c emits and what src/asahi/isa/AGX2.xml describes.
 *   - G15 (M3), G16 (M4), G17 (M5) and G18 (M6) are AGX3, a different
 *     instruction set. Apple's own compiler routes them through a separate
 *     LLVM target, and the community AGX2 decoder consumes 0% of their code.
 *   - AGX3 is not one encoding. Three binary encodings exist, and which one a
 *     part uses is now known for every chip this driver can enumerate.
 *
 * AGX3 now has separate decoders and encoders. Device support remains a
 * stricter question: G17P is admitted only through its exact T8140 contract,
 * while other AGX3 identities stay outside the runnable Honeykrisp path until
 * their independently-versioned pieces are connected.
 *
 * Evidence for every claim below is in research/gpu/, principally
 * GPU_AGX3_APPLE_ENCODER_TABLES_2026-08-23.md (encodings, Apple's tables),
 * GPU_DRIVER_CONTRACT_G15_G17_2026-08-23.md (the kernel core-name and chip-ID
 * decode that makes the mapping below well-founded) and
 * GPU_G15_G16_ISA_GENERATION_2026-08-23.md.
 */

enum agx_chip {
   /* AGX2. `G` is a single-cluster part, `X` is multi-cluster. */
   AGX_CHIP_G13G,
   AGX_CHIP_G13X,
   AGX_CHIP_G14G,
   AGX_CHIP_G14X,

   /* G15P is the A16 phone part and still executes AGX2. The Mac G15 parts
    * execute AGX3b; G15G additionally needs explicit scheduling nops. */
   AGX_CHIP_G15P,
   AGX_CHIP_G15G,
   AGX_CHIP_G15S,
   AGX_CHIP_G15C,
   AGX_CHIP_G15D,

   /* Every G16 part executes AGX3b. */
   AGX_CHIP_G16P,
   AGX_CHIP_G16G,
   AGX_CHIP_G16S,
   AGX_CHIP_G16C,

   /* G17P/G17A remain on the HAL200/AGX3b side of the boundary. The M5
    * G/S/C parts use AGX3c. */
   AGX_CHIP_G17P,
   AGX_CHIP_G17A,
   AGX_CHIP_G17G,
   AGX_CHIP_G17S,
   AGX_CHIP_G17C,

   /* The M6 base part (J873, T8152). Its command-stream records are the
    * M5 HAL300 ones, but its shaders use a third AGX3 encoding. */
   AGX_CHIP_G18G,

   AGX_CHIP_UNKNOWN,
};

/* Instruction set. Apple's compiler names these agx2 and agx3, and the low
 * nibble of an applegpu Mach-O cpusubtype is exactly this value. */
enum agx_isa {
   AGX_ISA_AGX2 = 2,
   AGX_ISA_AGX3 = 3,
   AGX_ISA_UNKNOWN = 0,
};

/*
 * Binary encoding.
 *
 * One instruction set, three encodings. The first two were established by
 * decoding a 274-kernel corpus compiled for the ten AGX3 families known at the
 * time with each of Apple's own encoder tables, recovered from
 * libapplegpu-nt.dylib. Every family decodes 100% under exactly one table and
 * 0-1.8% under any other -- no intermediate values:
 *
 *   agx3b (Apple variant 0): g15g, g15s, g15d, g16g, g16p, g16s, g17p
 *   agx3c (Apple variant 1): g17g, g17s, g18p
 *
 * The third is the macOS 27 toolchain's "g18g-encodings" subtarget feature,
 * next to "g17g-encodings":
 *
 *   agx3d (Apple variant 2): g18g, g18m, g19p
 *
 * Compiling one probe with metal-nt for every AGX3 arch gives byte-identical
 * code within that row and code unlike either row above, and 41 of 41 G18G
 * programs fail to decode under the agx3c, agx3b and AGX2 tables, where every
 * G17G one decodes cleanly under agx3c. The instruction set underneath is
 * still AGX3 (the applegpu_g18g Mach-O cpusubtype 0x1c3 has low nibble 3), and
 * roughly a fifth of the instructions keep their agx3c bytes, but the common
 * ALU, memory, control and texture forms are re-encoded with different
 * lengths. Its table is recovered (agx3_compose/agx3d_tables.c and
 * src/asahi/isa/AGX3d.xml, variant 2 of libapplegpu-nt.dylib), but only the
 * instruction forms a core encoder has grounded against Apple's G18G output
 * are emitted for it; see agx_chip_codegen_supported.
 *
 * An earlier partition by byte-identity read variant 0 as two groups,
 * splitting g15g off from the rest. That was an artefact: g15g uses the same
 * encoder table, and differs only in that its scheduler pads with nops. See
 * agx_chip_needs_scheduling_stalls.
 *
 * The names match src/asahi/isa/AGX3b.xml, AGX3c.xml and AGX3d.xml and the
 * disassembler entry points generated from them.
 */
enum agx_isa_encoding {
   AGX_ISA_ENCODING_AGX2,
   AGX_ISA_ENCODING_AGX3B,
   AGX_ISA_ENCODING_AGX3C,
   AGX_ISA_ENCODING_AGX3D,
   AGX_ISA_ENCODING_UNKNOWN,
};

/* Apple's independently-versioned command-stream/HAL axis.
 *
 * This names the layout of the records userspace writes -- VDM, PPP and CDM
 * words -- and not the kernel HAL that consumes them. G18G is served by
 * Apple's PI_400 kext, yet the records its Metal driver writes are the HAL300
 * ones; see agx_chip_hal300_records. */
enum agx_command_stream_generation {
   AGX_CS_GENERATION_AGX2,
   AGX_CS_GENERATION_G15,
   AGX_CS_GENERATION_HAL200,
   AGX_CS_GENERATION_HAL300,
   AGX_CS_GENERATION_UNKNOWN,
};

/* PBE/texture descriptor encoder generation. The descriptor grows from 24 to
 * 32 bytes at Gen9 and is laid out again at Gen10. */
enum agx_descriptor_generation {
   AGX_DESCRIPTOR_GENERATION_UNKNOWN = 0,
   AGX_DESCRIPTOR_GENERATION_7 = 7,
   AGX_DESCRIPTOR_GENERATION_8 = 8,
   AGX_DESCRIPTOR_GENERATION_9 = 9,
   AGX_DESCRIPTOR_GENERATION_10 = 10,
};

/* Samplers keep the Gen4 class through M4, but HAL200 moves fields within the
 * word. Gen5 adds reduction mode and a wider LOD clamp on M5, and G18G keeps
 * it. */
enum agx_sampler_generation {
   AGX_SAMPLER_GENERATION_GEN4,
   AGX_SAMPLER_GENERATION_GEN4_HAL200,
   AGX_SAMPLER_GENERATION_GEN5,
   AGX_SAMPLER_GENERATION_UNKNOWN,
};

/* M4 Pro/Max and A18 Pro carry both queueing paths; M5 and M6 are
 * SKSM-only. */
enum agx_submission_model {
   AGX_SUBMISSION_MODEL_CLASSIC,
   AGX_SUBMISSION_MODEL_DUAL,
   AGX_SUBMISSION_MODEL_SKSM,
   AGX_SUBMISSION_MODEL_UNKNOWN,
};

/* G17P keeps every colour target in the imageblock past 64 bytes a sample:
 * Apple's applegpu_g17p fragments store five RGBA32Float targets at tile
 * offsets 0..64 and eight at 0..112, and an eight-target mixed-format set at
 * 65 bytes, with no memory store, and its background builtins clear them at
 * the same offsets. */
#define AGX_G17P_MAX_BYTES_PER_SAMPLE (128)

/* On a G17P tile that no colour-writing geometry covered, the background
 * program -- clear or reload alike -- delivers only the low eight bytes of
 * each sample when the tilebuffer is multisampled. Measured on an A18 Pro:
 * with a sixteen-byte sample at 2x or 4x, such a texel reads back with its
 * upper eight bytes zero, while the same background program keeps the whole
 * sample on a tile a draw did cover -- including the pixels of that tile the
 * draw's scissor rejected. Eight-byte and narrower samples are unaffected at
 * every sample count, layer count and image size measured, and so is a
 * single-sample sixteen-byte target.
 *
 * Above this bound a load-op clear is emitted as a full-area draw instead,
 * which covers every tile and keeps the target in the tilebuffer. A reloaded
 * attachment has no such draw and spills. A multiview render has a second,
 * separate defect that the full-area draw does not cover --
 * dEQP-VK.multiview.multisample.no_queries.max_multi_view_view_count, 32
 * views of RGBA32F at 4x, whose clear value is all zeroes so the background
 * clear cannot be what it observes -- and spills too.
 */
#define AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE (8)

#ifndef __OPENCL_VERSION__
#include <stdbool.h>
#endif

/*
 * Whether native captures on this chip witness the HAL300 records Mesa's M5
 * packers write: the VDM state, PPP state and Index List words, the nine PPP
 * graphics records, the viewport, W-clamp/cull2, depth/stencil and
 * depth-bias/scissor PPP records, the CDM launch and barrier, and the ZLS
 * pixel dimensions.
 *
 * G17G (M5) is where these packers were grounded. G18G (M6) is served by
 * Apple's PI_400 kext rather than a PI_300 one, but for the thirteen J873
 * native modes (runs/mini-26A428/sessions-20260925-{a,b}: color, points,
 * lines, discard, blend-add, depth-less, depth-always, stencil, msaa4, r32f,
 * r8ui, texture-sample and compute-add) Apple's G18G records and Mesa's G17G
 * packers agree byte for byte. The one layout difference is the direct
 * non-indexed Index List, where G18G emits the tailed form; see
 * agx_vdm_index_list_nonindexed_has_tail. Lines, which M5 never captured,
 * are packed for G18G alone, and one stencil face bit is modelled for
 * neither chip; see agx_ppp_hal300_graphics_core.h.
 *
 * This is a statement about the record layouts only. G17S/G17C also write
 * HAL300 records but have no native graphics capture, so the codecs that
 * accept them list them by name. It says nothing about the shader encoding
 * or the kernel submission ABI.
 */
static inline bool
agx_chip_hal300_records(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G || chip == AGX_CHIP_G18G;
}

/*
 * Whether an encoding is one of the AGX3 tables (agx3b, agx3c or agx3d).
 *
 * For compiler decisions about the AGX3 instruction set as such -- which IR
 * shapes the AGX3 packers can take at all -- that were written as
 * "AGX3B || AGX3C" when those were the only AGX3 tables. It says nothing about
 * which forms a table has grounded encoders for; those gates stay per
 * encoding, and an ungrounded AGX3D form is refused by its core encoder.
 */
static inline bool
agx_encoding_is_agx3(enum agx_isa_encoding encoding)
{
   return encoding == AGX_ISA_ENCODING_AGX3B ||
          encoding == AGX_ISA_ENCODING_AGX3C ||
          encoding == AGX_ISA_ENCODING_AGX3D;
}

/*
 * Whether an encoding belongs to the G17-class machine: the M5 G17G/G17S/G17C
 * parts (agx3c) and the M6 G18G (agx3d).
 *
 * Use this for machine behaviour -- scoreboard slot counts, register and
 * uniform limits, the IR lowerings Apple's G17 compiler also performs -- and
 * never for byte layout. agx3c and agx3d share Apple's anonymous field ids and
 * operand semantics (the G18 scheduling-unit list in libapplegpu-nt.dylib is
 * the G17 one, MXU included; __GPU_ARCH_LD_MD is identical for 41 of 41
 * G17G/G18G metal-nt pairs; G17G field vectors composed with the variant-2
 * table reproduce 1,111 of 1,547 aligned G18G instructions, the rest being
 * compiler choices), but their encoding points, lengths and bit positions
 * differ for about 80% of emitted instructions. So a gate that selects agx3c
 * field ids, encoding points, lengths or hand-written bytes stays
 * AGX_ISA_ENCODING_AGX3C, and agx3d bytes come only from agx3_compose with
 * the agx3d table (agx3_bind_compose with AGX3_BIND_APPLE_ORDER).
 */
static inline bool
agx_encoding_is_g17_class(enum agx_isa_encoding encoding)
{
   return encoding == AGX_ISA_ENCODING_AGX3C ||
          encoding == AGX_ISA_ENCODING_AGX3D;
}

#ifndef __OPENCL_VERSION__

#include <stdbool.h>
#include <stdint.h>

#include "drm-uapi/asahi_drm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Return the exact command-stream architecture, or UNKNOWN.  Recognition does
 * not imply a complete Honeykrisp upload-to-submit path.  Callers opening a
 * device must check agx_chip_driver_supported, not only UNKNOWN. */
enum agx_chip
agx_chip_from_params(const struct drm_asahi_params_global *params);

/* The instruction set a chip executes. */
enum agx_isa agx_chip_isa(enum agx_chip chip);

/* The binary encoding of that instruction set. Known for every chip in the
 * enum; UNKNOWN only for AGX_CHIP_UNKNOWN. */
enum agx_isa_encoding agx_chip_isa_encoding(enum agx_chip chip);

enum agx_command_stream_generation
agx_chip_command_stream_generation(enum agx_chip chip);

enum agx_descriptor_generation
agx_chip_descriptor_generation(enum agx_chip chip);

enum agx_sampler_generation
agx_chip_sampler_generation(enum agx_chip chip);

/* Native sampler heap element size. The packed state word remains eight
 * bytes, but G15 and later place it at offset zero of a 32-byte sparse heap
 * element. Zero means the heap ABI is not proven for this identity. */
unsigned agx_chip_sampler_heap_element_size(enum agx_chip chip);

enum agx_submission_model
agx_chip_submission_model(enum agx_chip chip);

/*
 * Whether this driver can generate code for a chip.
 *
 * True for every encoding that has a packer: AGX2 through agx_pack.c,
 * AGX3b/AGX3c through agx3_pack_binary.c. False for AGX3d, whose table is
 * recovered and whose core encoders ground only a first instruction
 * catalogue, but for which agx3_pack_binary still refuses every shader (the
 * program-level pieces are not grounded), and for an unrecognised part, where
 * no packer can be chosen at all.
 *
 * This is a statement about the compiler, not about the driver. It says a
 * shader compiled for this chip is encoded by the right packer, and that an
 * ungrounded shader produces no binary rather than a mis-encoded one -- not
 * that the driver can submit the result. Device open must gate on
 * agx_chip_driver_supported.
 */
bool agx_chip_codegen_supported(enum agx_chip chip);

/* Whether every currently-required userspace axis is implemented. This is the
 * device-open gate; recognising a chip or having its ISA packer is not enough.
 */
bool agx_chip_driver_supported(enum agx_chip chip);

/* Validate the kernel-published identity as well as the userspace chip table.
 * G17P is the first identity whose runnable path depends on explicit USC and
 * HAL fields added to drm_asahi_params_global. */
bool agx_params_driver_supported(
   const struct drm_asahi_params_global *params);

/* Exact M5 kernel contract for explicitly opted-in driver integration.
 * This is not a declaration of a complete or conformant device path. */
bool agx_params_m5_bringup_supported(
   const struct drm_asahi_params_global *params);

/* Exact product name when the chip identity maps to one unambiguous SoC.
 * Variant letters are not marketing tiers: notably, G17P is A18 Pro rather
 * than an M5 variant. Returns NULL when one chip identity spans products. */
const char *agx_chip_exact_product_name(enum agx_chip chip);

/* libagx_g13{g,x} contains AGX2 machine code and USC records. It must never be
 * selected for an AGX3 device, even when no current draw is expected to call a
 * precompiled helper. */
bool agx_chip_uses_legacy_precompiled_programs(enum agx_chip chip);

/*
 * Hardware ray tracing.
 *
 * Present on every AGX3 part and on none before. Measured rather than
 * assumed: an intersect() call compiles to roughly 900-980 bytes on every
 * AGX3 family against 3315 bytes of software traversal on G13, G14 and the
 * one AGX2 G15 phone part.
 */
bool agx_chip_has_hardware_ray_tracing(enum agx_chip chip);

/*
 * Whether the compiler must insert explicit scheduling stalls.
 *
 * The M3 base part needs them and nothing else does: stripping its stall words
 * leaves an instruction stream 99.5% identical in count and 85.1%
 * byte-identical to M4's, with a further 7.4% differing only by branch
 * displacements shifted by exactly the removed padding.
 *
 * This is a property of one chip's scheduler, not of its encoding -- G15G
 * shares the agx3b encoder table with every other variant-0 part. It is a
 * codegen obligation, so it is only meaningful for a chip this driver can
 * generate code for.
 */
bool agx_chip_needs_scheduling_stalls(enum agx_chip chip);

/* Stable name for diagnostics, e.g. "G15X". Never NULL. */
const char *agx_chip_name(enum agx_chip chip);

/* Stable name for the instruction set, e.g. "AGX3". Never NULL. */
const char *agx_isa_name(enum agx_isa isa);

/* Stable name for the encoding, matching its XML table, e.g. "agx3b". Never
 * NULL. */
const char *agx_isa_encoding_name(enum agx_isa_encoding encoding);

const char *agx_command_stream_generation_name(
   enum agx_command_stream_generation generation);

#ifdef __cplusplus
}
#endif

#endif
