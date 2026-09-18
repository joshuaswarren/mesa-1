/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Strict AGX3 texel-fetch encoding.  This models the directly witnessed 2D,
 * 2D-array, 2D-MS, 2D-MS-array, and G17P 3D read forms. Unsupported texture
 * modes are represented by the absence of an API, rather than guessed
 * selector values.
 *
 * Native G17P filtered array samples select f77=1 and name a dynamic layer
 * register with f37=3/f31=<half-register>. This is distinct from the older
 * staged f77=2 shape and is witnessed for implicit, level, bias and gradient
 * LOD as well as immediate and uniform texture/sampler operands.
 *
 * Native G17P 2D texture offsets are carried directly in SMP: f141 enables
 * the offset, f51 selects its packed field, and f43 contains signed four-bit
 * x/y offsets in its low/high nibbles. The bounded form exposed here is the
 * immediate texture/sampler, implicit-LOD sample witnessed by Apple's own
 * compiler across both result widths and result masks.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Long enough for every emitted form: the texel fetch is 14 bytes (AGX3b) /
 * 8 bytes (AGX3c); the filtered sample is 14 bytes (AGX3b) / 22 bytes
 * (AGX3c, smp.ep3). */
#define AGX3_TEX_MAX_LENGTH 22u

enum agx3_tex_width {
   AGX3_TEX_W16 = 1,
   AGX3_TEX_W32 = 2,
};

/*
 * How a texture or sampler operand is located. The kind selects which of three
 * consecutive fields carries the value -- the same 1/2/3 convention documented
 * in agx3_copy_core.h -- and immediate vs non-immediate also selects the
 * encoding point (ep3 vs ep4 on AGX3b).
 *
 *     slot      immediate   register   uniform     selector
 *     texture   f95         f96        f97         f103
 *     sampler   f124        f125       f126        f132
 *
 * IMMEDIATE names an index into the texture state registers. REGISTER and
 * UNIFORM carry a zero-extended native heap index in a 64-bit value, located by half-register or
 * uniform-half index; Apple's own shaders load a 16-byte argument-buffer
 * entry and name the texture at +0 and the sampler at +8. All three kinds are
 * witnessed byte-exactly in test/test-agx3-smp-witness.c.
 */
enum agx3_tex_operand_kind {
   AGX3_TEX_OPERAND_IMMEDIATE = 1,
   AGX3_TEX_OPERAND_REGISTER = 2,
   AGX3_TEX_OPERAND_UNIFORM = 3,
};

/*
 * SMP's f83/f112 pair is part of a larger sample ABI, not a shader-stage
 * selector.  The driver's descriptor heaps, coordinate staging, and
 * derivative protocol are hardware-qualified with f83=0/f112=4.  Apple's
 * fragment binaries use f83=4/f112=8 together with a different, not yet
 * modeled surrounding ABI.
 *
 * Keep the Apple pair available to byte-exact witness tests, but require an
 * explicit full-ABI choice.  Production compiler callers leave this zero and
 * therefore cannot infer the Apple form merely from MESA_SHADER_FRAGMENT.
 */
enum agx3_tex_sample_abi {
   AGX3_TEX_SAMPLE_ABI_DRIVER = 0,
   AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT = 1,
};

/*
 * How the level operand staged into texture operand bank 6 is typed.
 *
 * PACK_SMPIR's f74 selects it, and it is the only field that moves between
 * two otherwise byte-identical fragment programs compiled by
 * `metal-nt -arch applegpu_g17p` from one source file: a filtered
 * `t.sample(s, uv, level(in.uv.z))` stages f74 = 2, while a
 * `t.read(uint2(in.uv.xy), uint(in.uv.z))` texel fetch stages f74 = 3.
 * Evidence: .tmp/g17p-gallium2-evidence/texlodtype.{metal,nat}.
 */
enum agx3_tex_level_kind {
   AGX3_TEX_LEVEL_MINIMUM = 1,
   AGX3_TEX_LEVEL_FLOAT = 2,
   AGX3_TEX_LEVEL_INTEGER = 3,
};

struct agx3_tex_fetch_desc {
   uint16_t dest;       /* half-register index */
   uint16_t coord;      /* four-half aligned pair of f32 coordinates */
   enum agx3_tex_width width;
   uint8_t mask;        /* Packed result channels, nonempty RGBA subset */
   uint8_t texture;     /* immediate texture index, compact-domain 0..15 */
   uint8_t slot;
   uint16_t wait_mask;  /* zero or one compactly encodable dependency */
   bool coord_kill;
   bool final;          /* witnessed final texture-op marker */
   bool array;          /* 2D array: layer in the register after the pair */
   bool multisample;    /* 2D MS read; sample/layer may be staged separately */
   bool staged_sample;  /* MS sample and optional array layer in staging half4 */
   bool three_d;        /* G17P 3D read: third contiguous integer coordinate */
   uint8_t sample;      /* immediate non-array sample index, values 0..3 */
   bool texture_kill;   /* Last use of an indirect texture index */
   /* Operand kinds. Zero is treated as IMMEDIATE so existing callers, which
    * set only `texture`, keep their exact previous encoding. */
   enum agx3_tex_operand_kind texture_kind;
   enum agx3_tex_operand_kind sampler_kind;
   /* Value for the non-immediate kinds; `texture` remains the immediate
    * index. Half-register index for REGISTER, uniform half for UNIFORM. */
   uint16_t texture_value;
   uint16_t sampler_value;
   /*
    * The level.
    *
    * By default the fetch is at level 0, which is what the immediate form
    * encodes with an all-zero f29. A non-zero `lod_x64` is the level in the
    * same 1/64 fixed-point field the sample family uses: compiling
    * `t.read(gid, 0)` against `t.read(gid, 2)` moves f29 from 0 to 128 and
    * nothing else.
    *
    * `register_lod` selects the per-lane form instead: f37 clears (there is
    * no immediate operand) and f139 becomes 4, with the level published into
    * operand bank 6 by an agx3_tex_stage_lod() carrying
    * AGX3_TEX_LEVEL_INTEGER. Those two fields are the ONLY difference
    * between the two fragment texel fetches in
    * .tmp/g17p-gallium2-evidence/texlodfs.{metal,nat}. AGX3b only; AGX3c has
    * no witness.
    */
   uint16_t lod_x64;
   bool register_lod;
   /*
    * Sparse residency (VK_KHR_sparse_*, OpImageSparse*).  The instruction
    * writes its masked result channels as usual and then one extra 32-bit
    * register holding the residency code, at the half immediately after the
    * masked channels -- the same destination convention AGX2 uses.
    *
    * These fields were isolated by compiling matched pairs that differ
    * only in `sparse_sample`/`sparse_read` vs `sample`/`read`, with
    * `xcrun metal-nt -arch applegpu_g17p`
    * (MetalToolchain v27.1.5194.15, -platform_version macos 27.0 27.0):
    *
    *   f142 = 1     the residency request itself
    *   f120 = 1     set on every sparse access
    *   f112 += 4    NOT ENCODED.  Apple's bytes do move the operand profile,
    *                but f112 is a scattered bitmask rather than a counter, so
    *                "+4" drops profile bit 2 and sets bit 3 -- and on G17P
    *                that makes every sparse fetch and every explicit-LOD
    *                sparse sample fail.  The profile stays at its dense
    *                value; see the note in agx3_tex_encode_sample() for the
    *                measurement.
    *
    * Matched compute pairs, argument-buffer descriptors on both arms
    * (test/probes/g17p-sparse-residency.metal, k_ab_dense vs k_ab_sparse and
    * k_ab_dread vs k_ab_sread), differ in exactly those three fields plus
    * register allocation and the group terminator (the f112 move among them,
    * which this encoder deliberately does not reproduce):
    *
    *   sample  dense  05800cb8b4000900920390010100802000800098
    *           sparse 2dc00c38bc000900920392010200802028c00018
    *   fetch   dense  05800cb8b0001701900280000100002000800098
    *           sparse 2dc00c38b0001701900282000200002028c00018
    *
    * The residency register is read straight back by the `bit_andc` that
    * follows: with a W32 result at half 10 and mask 15 it names half 18, with
    * a W32 result at half 0 and mask 1 it names half 2, and with a W16 result
    * at half 0 and mask 15 it names half 4 -- dest + channels * halves.
    *
    * f120 deposits instruction bit 142, past the 112-bit ep3 form, so the
    * composer refuses a sparse ep3 instruction outright and a sparse
    * operation is always the 20-byte ep4 form -- even when both descriptors
    * are immediate, where the dense sibling stays 14 bytes.  (f142 itself
    * deposits bit 81 and would fit either form.)  Only AGX3b is witnessed.
    */
   bool sparse;
};

bool agx3_tex_encode_fetch(enum agx_isa_encoding enc,
                           const struct agx3_tex_fetch_desc *d,
                           uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* Strict canonical decode: trailing bytes, alternate forms and any bit pattern
 * that does not repack byte-for-byte through the bounded encoder are rejected.
 */
bool agx3_tex_decode_fetch(enum agx_isa_encoding enc, const void *instruction,
                           uint32_t instruction_size,
                           struct agx3_tex_fetch_desc *out);

/*
 * Filtered 2D sample (`texture_sample` -> smp with f78=2), immediate texture
 * and sampler state, in the LOD shapes Apple's corpus witnesses:
 *
 *  - AGX3_TEX_LOD_AUTO: the implicit-LOD fragment form (no LOD operand).
 *    Witnessed in fragment render pipelines only; fragment shaders are the
 *    only place Mesa emits it, matching the hardware's quad-derivative
 *    requirement. f156 terminates the complete fragment sample group, not
 *    each individual implicit sample; see sample_group_end below.
 *  - AGX3_TEX_LOD_LEVEL: explicit immediate level (f37=1, f139=2), the
 *    `level(x)` kernel form; `lod_x64` is the level in 1/64 steps exactly as
 *    the corpus quantizes it (level(2.0) = 128, level(7.25) = 464).
 *  - AGX3_TEX_LOD_REGISTER: G17P fragment per-lane half-float level, packed
 *    by PACK_SMPIR and selected with f139=4. The fragment marker is also
 *    present for explicit LOD; it does not itself select implicit derivatives.
 *
 * G17P also supports directly witnessed 2D-array forms with a register-valued
 * layer for implicit, level, bias and gradient LOD, and cube samples with
 * implicit or explicit immediate/register LOD.
 * G17P 3D immediate-level samples use the same operand ABI, with f76=3
 * selecting the third contiguous coordinate.
 * Everything else about the sampler family -- remaining dimensions, min-LOD,
 * offset shapes beyond witnessed 2D sampling, 3D/cube shadow/gather/gradient,
 * and array combinations -- remains a deliberate refusal: those axes need
 * additional operand marshalling and qualification.
 */
enum agx3_tex_lod {
   AGX3_TEX_LOD_AUTO,
   AGX3_TEX_LOD_LEVEL,
   AGX3_TEX_LOD_REGISTER, /* PACK_SMPIR publishes the per-lane level. */
   /*
    * Implicit LOD plus a constant bias.  This is the ordinary automatic-LOD
    * sample with f139 = 1 and the bias in the same 12-bit x64 immediate the
    * explicit-level form uses, witnessed from `metal-nt -arch applegpu_g17p`
    * on t.sample(s, uv, bias(k)) for k in {0.25, 0.5, 1, 1.5, 2, 4, 1/64, 31}:
    * f29 came out 16, 32, 64, 96, 128, 256, 1, 1984 -- exactly k * 64, and
    * 63.0 saturated the field at 2047.  A zero bias is folded away by Apple's
    * own compiler into the plain implicit form.
    */
   AGX3_TEX_LOD_BIAS,
   /*
    * Implicit LOD plus a per-lane register bias.  Apple's fragment
    * `t.sample(s, uv, bias(in.uv.x))` is byte-identical to its
    * `t.sample(s, uv, level(in.uv.x))` apart from f139 (3 instead of 4): the
    * same PACK_SMPIR scalar staging into operand bank 6, the same fragment
    * marker f156 = 1.  Ground truth, `xcrun metal-nt -arch applegpu_g17p`:
    *
    *    level(in.uv.x)  05800cb8b0000001010010001200   f139 = 4
    *    bias(in.uv.x)   05800cb8b0000400010010001200   f139 = 3
    */
   AGX3_TEX_LOD_BIAS_REGISTER,
   /*
    * Explicit gradients.  f139 = 5, and unlike every other shape the
    * fragment marker f156 is *not* set: the four gradient scalars replace
    * the derivative the marker would request.  The scalars are published
    * into staging banks 8/10/12/14, in half-register units, before the
    * sample.  Apple's own compute kernel
    * `t.sample(s, g[0], gradient2d(g[1], g[2]))` stages six consecutive
    * source registers into banks 0, 2, 8, 10, 12, 14 on one shared slot and
    * then samples:
    *
    *    copy 0b10210c  bank  0     copy 5b16210c  bank 10
    *    copy 1b12210c  bank  2     copy 6b18210c  bank 12
    *    copy 4b14210c  bank  8     copy 7b1a210c  bank 14
    *    smp  05800cb8b0000401000010000100
    *
    * Which component lands in which bank is pinned by the fragment kernels
    * that pass literal gradients, where the staged value is the constant
    * itself: gradient2d((0.25, 0.5), (0.75, 1.0)) loads 0.25/0.75/0.5/1.0
    * into banks 8/10/12/14, so the order is (ddx.x, ddy.x, ddx.y, ddy.y).
    * The (1, 2)/(4, 8) and (16, 32)/(64, 128) pairs agree.  That is exactly
    * the layout of the collect agx_emit_tex() builds for nir_tex_src_ddx.
    */
   AGX3_TEX_LOD_GRADIENT,
};

/*
 * Staging banks for AGX3_TEX_LOD_GRADIENT, in half-register units, in the
 * order the four scalars are witnessed.  A gradient sample also names
 * coordinate bank 0, as every witness does.
 */
#define AGX3_TEX_GRADIENT_BANK_DDX_X 8u
#define AGX3_TEX_GRADIENT_BANK_DDY_X 10u
#define AGX3_TEX_GRADIENT_BANK_DDX_Y 12u
#define AGX3_TEX_GRADIENT_BANK_DDY_Y 14u
/* A gradient3d continues the same interleave into two more banks. Read off
 * two literal-gradient kernels, where each staged value is the constant
 * itself: gradient3d((1,2,4),(8,16,32)) puts 1/8/2/16/4/32 into banks
 * 8/10/12/14/16/18, and gradient3d((64,128,0.25),(0.5,0.75,1.5)) puts
 * 64/0.5/128/0.75/0.25/1.5 into the same six. */
#define AGX3_TEX_GRADIENT_BANK_DDX_Z 16u
#define AGX3_TEX_GRADIENT_BANK_DDY_Z 18u

/*
 * textureGather.  f144 = 1 selects the gather and f145 carries the component
 * index -- absent for R, 1 for G, 2 for B, 3 for A.  A gather also *drops*
 * the f74 mask-group bit that every ordinary sample sets; nothing else about
 * the instruction changes, so a gather is a plain implicit sample with one
 * bit cleared and one or two set.  Ground truth,
 * `xcrun metal-nt -arch applegpu_g17p`, five fragment kernels differing only
 * in the gathered component:
 *
 *    t.gather(s, uv)                       05800ca490000000010000001200
 *    t.gather(s, uv, int2(0), component::x) 05800ca490000000010000001200  f145 -
 *    t.gather(s, uv, int2(0), component::y) 05800cac90000000010000001200  f145 1
 *    t.gather(s, uv, int2(0), component::z) 05800cb490000000010000001200  f145 2
 *    t.gather(s, uv, int2(0), component::w) 05800cbc90000000010000001200  f145 3
 *
 * 2D-array and cube gathers share the same component selector. Runtime
 * 2D offsets use f141=2 with the bank-6 publication declared below; immediate
 * offsets retain f43/f51/f141. Gathers take no explicit LOD argument.
 */
enum agx3_tex_gather {
   AGX3_TEX_GATHER_NONE = 0,
   AGX3_TEX_GATHER_R,
   AGX3_TEX_GATHER_G,
   AGX3_TEX_GATHER_B,
   AGX3_TEX_GATHER_A,
};

struct agx3_tex_sample_desc {
   uint16_t dest;       /* half-register index */
   uint16_t coord;      /* four-half aligned pair of f32 coordinates */
   enum agx3_tex_width width;
   uint8_t mask;        /* G17P: every nonempty RGBA subset, packed results */
   uint8_t texture;     /* immediate texture-state index, 0..15 */
   uint8_t sampler;     /* immediate sampler-state index, 0..15 */
   enum agx3_tex_lod lod;
   /* G17P f140=2 consumes a native bank-6 minimum, combined with the
    * half-float bias or published beside explicit gradients. */
   bool min_lod;
   /* A 2D-array comparison with a bias reads its minimum from the GPR
    * agx3_tex_pack_minimum() fills instead (f37=2, f30, f41). */
   bool min_lod_register;
   uint16_t min_lod_value;
   bool min_lod_kill;
   uint16_t lod_x64;    /* level * 64 when LEVEL, bias * 64 when BIAS */
   uint8_t slot;
   uint16_t wait_mask;  /* zero or one compactly encodable dependency */
   bool coord_kill;
   bool final;          /* witnessed final texture-op marker */
   /* f156 terminates a fragment sample group. Fresh G17P two/three-sample
    * pipelines clear it on every earlier SMP and set it only on the final
    * ordinary sample. Explicit-gradient samples sit outside the group and
    * reject both flags. These explicit aggregate-group flags are G17P-only;
    * leaving both clear preserves the independently witnessed per-op encoding
    * in every other stage/ISA. */
   bool sample_group_nonterminal;
   bool sample_group_end;
   /* Vertex explicit-LOD samples use the same SMP fields and operand staging
    * as fragment samples, but never carry fragment group terminator f156. */
   bool vertex_stage;
   /* Compute samples likewise stage exactly as fragment samples do, and
    * likewise never carry f156. */
   bool compute_stage;
   bool array;           /* array sample; shadow layers use PACK_SMPIR */
   uint16_t array_layer; /* 32-bit layer half-register, even and <=254 */
   bool three_d;        /* 3D level: third contiguous f32 coordinate, G17P */
   bool cube;           /* native cube; combine with array for cube-array */
   enum agx3_tex_operand_kind texture_kind;
   enum agx3_tex_operand_kind sampler_kind;
   uint16_t texture_value;
   uint16_t sampler_value;
   bool texture_kill;
   bool sampler_kill;
   enum agx3_tex_gather gather; /* textureGather component, or NONE */
   bool shadow;                 /* compare against the scalar in bank 4 */
   /* G17P f146 queries LOD, with f74 clear. X is clamped, Y unclamped.
    * Array queries take no layer and encode immediate layer zero. */
   bool query_lod;
   bool offset;                 /* coordinate offset is present */
   bool offset_staged;          /* f141=2 reads PACK_SMPIR offset banks */
   bool offset_staged_array;    /* staged word also contains array layer */
   uint16_t offset_xyz;         /* immediate signed x/y/z nibbles */
   bool offset_register;        /* runtime gather offset staged in bank 6 */
   enum agx3_tex_sample_abi sample_abi;
   /* See agx3_tex_fetch_desc::sparse; the sample form uses the same fields,
    * witnessed on 2D, 2D-array, 3D, cube, gather, immediate level and bias
    * shapes and on both result widths, and likewise leaves the f112 operand
    * profile alone. */
   bool sparse;
};

/* Pack a scalar LOD into native texture operand bank 6. This bank is separate
 * from GPRs and the coordinate staging bank. `kind` selects how the level is
 * typed: FLOAT for the filtered sample's 16-bit `level(x)`, INTEGER for a
 * texel fetch's 32-bit uint level. `src` is a half-register address in either
 * case, so integer callers must name an aligned 32-bit register. */
bool agx3_tex_stage_typed_lod(enum agx_isa_encoding enc, uint16_t src,
                              uint8_t slot, uint16_t wait_mask,
                              bool src_cache, bool src_discard,
                              enum agx3_tex_level_kind kind,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* The `_at` stagers publish the operands of a read whose coordinates are
 * staged at half `bank` (0 or 16) into bank + 6 instead of bank 6. */
/* CEM's selected face together with a per-lane level or bias: the same
 * two-operand word as agx3_tex_stage_array_lod with the cube selector.
 * Apple's `depthcube.sample_compare(s, dir, ref, level(v))` fragment emits
 * exactly 1705540600040c00f0c01100 for discarded face r1 and level half 3. */
bool agx3_tex_stage_cube_z_lod(enum agx_isa_encoding enc, uint16_t z,
                               uint16_t lod, uint8_t slot, uint16_t wait_mask,
                               bool z_cache, bool z_discard, bool lod_cache,
                               bool lod_discard,
                               uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);
/* Cube-array comparison with a per-lane level or bias: the face+level word
 * above with the three cube-array selectors (f51 = 2, f55 = 1, f75 = 12).
 * Apple's `depthcube_array.sample_compare(s, dir, layer, ref, level(v))`
 * fragment emits exactly 1705540600040c00f0c21700 for face r1 and level
 * half 3, both discarded, on slot 1. */
bool agx3_tex_stage_cube_array_z_lod(enum agx_isa_encoding enc, uint16_t z,
                                     uint16_t lod, uint8_t slot,
                                     uint16_t wait_mask, bool z_cache,
                                     bool z_discard, bool lod_cache,
                                     bool lod_discard,
                                     uint8_t out[AGX3_TEX_MAX_LENGTH],
                                     uint8_t *len);
/* The auxiliary integer layer publication that accompanies that word: the
 * array-layer PACK_SMPIR (f74 = 4) with f5 = 0, f9 = 4 and three fixed bits
 * the binder does not model. Apple emits 1705540003200000f0880400 for a
 * discarded layer at half 16 on slot 1; its uniform-level preambles use the
 * same f5/f9/fixed-bit pattern with the FLOAT kind. `src` is a half index. */
bool agx3_tex_stage_aux_layer(enum agx_isa_encoding enc, uint16_t src,
                              uint8_t slot, uint16_t wait_mask,
                              bool src_cache, bool src_discard,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);
/* The same publication into a rotated staging group: `bank` is 6, 14 or 22,
 * the level operand of groups 0, 8 and 16. Apple's second in-flight fragment
 * texel fetch stages its level at bank 14 (PACK_SMPIR f5 = 14). */
bool agx3_tex_stage_typed_lod_bank(enum agx_isa_encoding enc, uint16_t bank,
                                   uint16_t src, uint8_t slot,
                                   uint16_t wait_mask, bool src_cache,
                                   bool src_discard,
                                   enum agx3_tex_level_kind kind,
                                   uint8_t out[AGX3_TEX_MAX_LENGTH],
                                   uint8_t *len);

bool agx3_tex_stage_typed_lod_at(enum agx_isa_encoding enc, uint16_t bank,
                                 uint16_t src, uint8_t slot,
                                 uint16_t wait_mask, bool src_cache,
                                 bool src_discard,
                                 enum agx3_tex_level_kind kind,
                                 uint8_t out[AGX3_TEX_MAX_LENGTH],
                                 uint8_t *len);

bool agx3_tex_stage_bias_min(enum agx_isa_encoding enc, uint16_t bias,
                            uint16_t minimum, uint8_t slot, uint16_t wait_mask,
                            bool src_cache, bool src_discard,
                            uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* The float-level form, which is what the filtered sample path stages. */
bool agx3_tex_stage_lod(enum agx_isa_encoding enc, uint16_t src, uint8_t slot,
                       uint16_t wait_mask, bool src_cache, bool src_discard,
                       uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* Pack the dynamic layer used by array fetches and driver-ABI samples. */
bool agx3_tex_stage_gather_offset(enum agx_isa_encoding enc, uint16_t src,
                                   bool array, uint16_t packed_layer,
                                   uint8_t slot, bool cache, bool discard,
                                   uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* PACK_IMGWIR publishes an image write's layer register to staging half
 * `dest` (6 in bank 0), with the same operand fields as PACK_SMPIR's array
 * layer. */
bool agx3_tex_stage_image_write_layer(enum agx_isa_encoding enc, uint16_t src,
                                      uint16_t dest,
                                      uint8_t slot, uint16_t wait_mask,
                                      bool src_cache, bool src_discard,
                                      uint8_t out[AGX3_TEX_MAX_LENGTH],
                                      uint8_t *len);

bool agx3_tex_stage_array_layer(enum agx_isa_encoding enc, uint16_t src,
                                uint8_t slot, uint16_t wait_mask,
                                bool src_cache, bool src_discard,
                               uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);
bool agx3_tex_stage_array_layer_at(enum agx_isa_encoding enc, uint16_t bank,
                                   uint16_t src, uint8_t slot,
                                   uint16_t wait_mask, bool src_cache,
                                   bool src_discard,
                                   uint8_t out[AGX3_TEX_MAX_LENGTH],
                                   uint8_t *len);

/* Publish the layer and per-lane half-float LOD together into operand bank 6. */
bool agx3_tex_stage_array_lod(enum agx_isa_encoding enc, uint16_t layer,
                              uint16_t lod, uint8_t slot, uint16_t wait_mask,
                              bool layer_cache, bool layer_discard,
                              bool lod_cache, bool lod_discard,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* Publish a 2D-array bias sample's minimum LOD at half 4, beside the layer and
 * bias staged in bank 6. */
bool agx3_tex_stage_array_minimum_lod(enum agx_isa_encoding enc, uint16_t src,
                                      uint8_t slot, uint16_t wait_mask,
                                      bool src_cache, bool src_discard,
                                      uint8_t out[AGX3_TEX_MAX_LENGTH],
                                      uint8_t *len);

/* Publish a 2D-array gradient sample's layer and minimum LOD into bank 6. */
bool agx3_tex_stage_array_minimum_gradient(
   enum agx_isa_encoding enc, uint16_t layer, uint16_t minimum, uint8_t slot,
   uint16_t wait_mask, bool layer_cache, bool layer_discard,
   bool minimum_cache, bool minimum_discard,
   uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* The texel-fetch variant packs a full uint LOD beside the array layer. */
bool agx3_tex_stage_array_integer_lod(
   enum agx_isa_encoding enc, uint16_t layer, uint16_t lod, uint8_t slot,
   uint16_t wait_mask, bool layer_cache, bool layer_discard, bool lod_cache,
   bool lod_discard, uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);
bool agx3_tex_stage_array_integer_lod_at(
   enum agx_isa_encoding enc, uint16_t bank, uint16_t layer, uint16_t lod,
   uint8_t slot, uint16_t wait_mask, bool layer_cache, bool layer_discard,
   bool lod_cache, bool lod_discard, uint8_t out[AGX3_TEX_MAX_LENGTH],
   uint8_t *len);

/* Publish a runtime signed 2D coordinate offset into operand bank 6. */
bool agx3_tex_stage_offset_2d(enum agx_isa_encoding enc, uint16_t x,
                              uint16_t y, uint8_t slot, uint16_t wait_mask,
                              bool source_cache, bool source_discard,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* Run one axis of the native cube-coordinate calculation.  Three axes with
 * the fixed reciprocal between axis 0 and 1 are required before sampling. */
bool agx3_tex_cube_cem_calc(enum agx_isa_encoding enc, uint16_t x,
                            uint16_t y, uint16_t z, uint8_t axis,
                            bool discard,
                            uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);
bool agx3_tex_cube_cem_calc_to(enum agx_isa_encoding enc, uint16_t x,
                               uint16_t y, uint16_t z, uint16_t dest,
                               uint8_t axis, bool discard,
                               uint8_t out[AGX3_TEX_MAX_LENGTH],
                               uint8_t *len);

bool agx3_tex_cube_cem_rcp(enum agx_isa_encoding enc,
                           uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);
bool agx3_tex_cube_cem_rcp_to(enum agx_isa_encoding enc, uint16_t value,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* Finish CEM's two normalized coordinates. `component` is 0 then 1. */
bool agx3_tex_cube_cem_finalize(enum agx_isa_encoding enc, uint8_t component,
                                bool discard,
                                uint8_t out[AGX3_TEX_MAX_LENGTH],
                                uint8_t *len);
bool agx3_tex_cube_cem_finalize_to(
   enum agx_isa_encoding enc, uint8_t component, uint16_t reciprocal,
   uint16_t axis_value, uint8_t slot,
   uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* Publish the face calculated by CEM for a native cube sample. */
bool agx3_tex_stage_cube_z(enum agx_isa_encoding enc, uint16_t src,
                           uint8_t slot, uint16_t wait_mask,
                           bool src_cache, bool src_discard,
                           uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* Publish CEM's face at half `face` together with the 16-bit cube-array
 * layer at half `layer`. */
bool agx3_tex_stage_cube_array_layer(enum agx_isa_encoding enc, uint16_t face,
                                     uint16_t layer, uint8_t slot,
                                     uint16_t wait_mask, bool layer_cache,
                                     bool layer_discard,
                                     uint8_t out[AGX3_TEX_MAX_LENGTH],
                                     uint8_t *len);

/* Publish CEM's face with a 16-bit float register level or bias. The face
 * half is always consumed. */
bool agx3_tex_stage_cube_lod(enum agx_isa_encoding enc, uint16_t face,
                             uint16_t lod, uint8_t slot, uint16_t wait_mask,
                             bool lod_cache, bool lod_discard,
                             uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* Publish CEM's face, a 16-bit float register level or bias and a cube-array
 * layer already packed into the 32-bit GPR `packed_layer` by
 * agx3_tex_pack_ms_array_layer(). The face and the packed word are always
 * consumed. */
bool agx3_tex_stage_cube_array_lod(enum agx_isa_encoding enc, uint16_t face,
                                   uint16_t lod, uint16_t packed_layer,
                                   uint8_t slot, uint16_t wait_mask,
                                   bool lod_cache, bool lod_discard,
                                   uint8_t out[AGX3_TEX_MAX_LENGTH],
                                   uint8_t *len);

/* Publish a register-valued integer 2D-MS sample index into the sampler
 * operand bank. The following MS fetch uses the sample-0 instruction form;
 * the staged value selects the sample. */
bool agx3_tex_stage_ms_sample(enum agx_isa_encoding enc, uint16_t src,
                              uint16_t wait_mask, bool src_cache,
                              bool src_discard,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* Pack CEM's face alone into the 32-bit GPR `dest` for a biased cube sample
 * with a minimum LOD. The face half is always consumed. */
bool agx3_tex_pack_cube_face(enum agx_isa_encoding enc, uint16_t face,
                             uint16_t dest, uint8_t out[AGX3_TEX_MAX_LENGTH],
                             uint8_t *len);

/* Publish a cube sample's minimum LOD and bias into bank 6 beside the face
 * agx3_tex_pack_cube_face() packed into `packed_face`. */
bool agx3_tex_stage_cube_bias_min(enum agx_isa_encoding enc, uint16_t bias,
                                  uint16_t minimum, uint16_t packed_face,
                                  uint8_t slot, uint16_t wait_mask,
                                  bool src_cache, bool src_discard,
                                  uint8_t out[AGX3_TEX_MAX_LENGTH],
                                  uint8_t *len);

/* Pack a 16-bit float minimum LOD into the 32-bit GPR `dest` for a 2D-array
 * comparison sample, which stages its reference in half 4. */
bool agx3_tex_pack_minimum(enum agx_isa_encoding enc, uint16_t src,
                           uint16_t dest, bool cache, bool discard,
                           uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

bool agx3_tex_pack_ms_array_layer(enum agx_isa_encoding enc, uint16_t src,
                                 uint16_t dest, bool cache, bool discard,
                                 uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);
bool agx3_tex_stage_ms_array_sample(enum agx_isa_encoding enc, uint16_t sample,
                                   uint16_t packed_layer, bool cache,
                                   bool discard,
                                   uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

bool agx3_tex_encode_sample(enum agx_isa_encoding enc,
                            const struct agx3_tex_sample_desc *d,
                            uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/*
 * The scoreboard-signalling coordinate-staging copy Apple issues before smp.
 *
 * Every witnessed smp reads its coordinates through registers that a
 * signalling producer published on the smp's wait slot. When the coordinates
 * do not come from an async op directly, Apple stages them with this `copy`
 * form -- one 32-bit register per instruction, dest == src when the values
 * are already in place (x_coordreg, fs_bufcoord) -- carrying the smp's slot
 * in f12 and any producer dependency as a wait bit (f30 + slot - 1). The smp
 * then waits on that slot.
 */
#define AGX3_TEX_COPY_MAX_LENGTH 10u

/* Native G17P image-write operand description. Coordinates and data are
 * staged by the caller. wait_mask describes the upstream dependency consumed
 * by its first data COPY; the explicit encoder receives the staging slot
 * separately. IMMEDIATE and REGISTER select native descriptor-index forms.
 * The existing native-witness wrapper retains its fixed base/slot choices. */
#define AGX3_IMAGE_WRITE_LENGTH 16u

/* The texture type an image write addresses. A layered write stages its
 * layer through agx3_tex_stage_image_write_layer(); a 3D write stages z as a
 * third coordinate. */
enum agx3_image_write_dim {
   AGX3_IMAGE_WRITE_2D = 0,
   AGX3_IMAGE_WRITE_2D_ARRAY,
   AGX3_IMAGE_WRITE_3D,
};

struct agx3_image_write_desc {
   uint16_t wait_mask;
   /* Staging half of the first coordinate (f5), a multiple of 8, and the
    * slot this imgwci signals when done (f12); a zero slot means slot 1. */
   uint16_t staging_bank;
   uint8_t result_slot;
   enum agx3_tex_operand_kind image_kind;
   uint16_t image_value;
   bool image_kill;
   bool normalized;
   enum agx3_image_write_dim dim;
};

/* Explicit PBE heap base in uniform halves, native descriptor index in
 * image_value, and the producer slot of the preceding staging COPYs. */
bool agx3_tex_encode_image_write_2d(
   enum agx_isa_encoding enc, const struct agx3_image_write_desc *d,
   uint16_t uniform_base, uint8_t staging_slot,
   uint8_t out[AGX3_IMAGE_WRITE_LENGTH], uint8_t *len);

bool agx3_tex_encode_image_write_rgba8_2d(
   enum agx_isa_encoding enc, const struct agx3_image_write_desc *d,
   uint8_t out[AGX3_IMAGE_WRITE_LENGTH], uint8_t *len);

/* A native staging COPY must preserve a coordinate GPR if later work uses
 * it. CACHE/NONE/DISCARD are independent of the staging-bank release bit. */
bool agx3_tex_stage_copy_hints(enum agx_isa_encoding enc, uint16_t dest,
                               uint16_t src, uint8_t slot, uint16_t wait_mask,
                               bool src_cache, bool src_discard,
                               uint8_t out[AGX3_TEX_COPY_MAX_LENGTH], uint8_t *len);

bool agx3_tex_stage_copy(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t src, uint8_t slot, uint16_t wait_mask,
                         uint8_t out[AGX3_TEX_COPY_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
