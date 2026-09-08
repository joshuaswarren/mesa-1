/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Strict AGX3 2D texel-fetch encoding.  This deliberately models only the
 * directly witnessed LOD0 forms with immediate texture selection and sampler
 * zero.  Unsupported texture modes are represented by the absence of an API,
 * rather than guessed selector values.
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
 *  - AGX3_TEX_LOD_AUTO: the implicit-LOD fragment form (f156=1, no LOD
 *    operand). Witnessed in fragment render pipelines only; fragment
 *    shaders are the only place Mesa emits it, matching the hardware's
 *    quad-derivative requirement.
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
 * Every witness is a plain 2D implicit-LOD gather: no array layer, no LOD
 * argument and no coordinate offset, so those stay refused.
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
   uint16_t lod_x64;    /* level * 64 when LEVEL, bias * 64 when BIAS */
   uint8_t slot;
   uint16_t wait_mask;  /* zero or one compactly encodable dependency */
   bool coord_kill;
   bool final;          /* witnessed final texture-op marker */
   bool array;          /* 2D array: layer is named directly by f31/f37 */
   uint16_t array_layer; /* 32-bit layer half-register, even and <=254 */
   bool three_d;        /* 3D level: third contiguous f32 coordinate, G17P */
   bool cube;           /* native cube: third coordinate in cube operand bank */
   enum agx3_tex_operand_kind texture_kind;
   enum agx3_tex_operand_kind sampler_kind;
   uint16_t texture_value;
   uint16_t sampler_value;
   bool texture_kill;
   bool sampler_kill;
   enum agx3_tex_gather gather; /* textureGather component, or NONE */
   bool shadow;                 /* compare against the scalar in bank 4 */
   bool offset;                 /* signed nibble-packed 2D texel offset */
   uint8_t offset_xy;           /* x in bits 3:0, y in bits 7:4 */
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

/* The float-level form, which is what the filtered sample path stages. */
bool agx3_tex_stage_lod(enum agx_isa_encoding enc, uint16_t src, uint8_t slot,
                       uint16_t wait_mask, bool src_cache, bool src_discard,
                       uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* Pack the dynamic layer/sample word used by array texel fetches into the
 * separately witnessed PACK_SMPIR form. Filtered G17P array samples instead
 * name their layer register directly in SMP. */
bool agx3_tex_stage_array_layer(enum agx_isa_encoding enc, uint16_t src,
                                uint8_t slot, uint16_t wait_mask,
                                bool src_cache, bool src_discard,
                                uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* Run one axis of the native cube-coordinate calculation.  Three axes with
 * the fixed reciprocal between axis 0 and 1 are required before sampling. */
bool agx3_tex_cube_cem_calc(enum agx_isa_encoding enc, uint16_t x,
                            uint16_t y, uint16_t z, uint8_t axis,
                            bool discard,
                            uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

bool agx3_tex_cube_cem_rcp(enum agx_isa_encoding enc,
                           uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* Publish the face calculated by CEM for a native cube sample. */
bool agx3_tex_stage_cube_z(enum agx_isa_encoding enc, uint16_t src,
                           uint8_t slot, uint16_t wait_mask,
                           bool src_cache, bool src_discard,
                           uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len);

/* Publish a register-valued integer 2D-MS sample index into the sampler
 * operand bank. The following MS fetch uses the sample-0 instruction form;
 * the staged value selects the sample. */
bool agx3_tex_stage_ms_sample(enum agx_isa_encoding enc, uint16_t src,
                              uint16_t wait_mask, bool src_cache,
                              bool src_discard,
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

/*
 * Exact G17P storage-image write instruction from native direct and
 * argument-buffer-indexed `texture2d<..., access::write>` witnesses. The
 * instruction consumes fixed coordinate/data staging lanes populated by the
 * caller and waits on staging slot 2. `wait_mask` is the upstream texture
 * dependency consumed by the first data-staging write, not an imgwci field.
 *
 * IMMEDIATE names the two descriptor-specialized states already used by the
 * public transfer paths: state 0/UINT and state 1/UNORM. REGISTER names an
 * even 32-bit GPR containing a native argument-buffer heap index; only the
 * normalized form is witnessed. UNIFORM remains deliberately unsupported.
 */
#define AGX3_IMAGE_WRITE_LENGTH 16u

struct agx3_image_write_desc {
   uint16_t wait_mask;
   enum agx3_tex_operand_kind image_kind;
   uint16_t image_value;
   bool image_kill;
   bool normalized;
};

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
