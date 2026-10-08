/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 texture encoding: texel fetches (2D, 2D-array, 2D-MS, 2D-MS-array,
 * G17P 3D), filtered samples and the operand-staging instructions they read.
 * Texture modes without an entry point here are refused, not guessed.
 *
 * G17P filtered array samples set f77=1 and name the layer register with
 * f37=3/f31 (half-register). Immediate 2D offsets are carried in SMP itself:
 * f141 enables them, f51 selects the packed field and f43 holds signed 4-bit
 * x/y offsets in its low/high nibbles.
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
 * witnessed in Apple's output.
 */
enum agx3_tex_operand_kind {
   AGX3_TEX_OPERAND_IMMEDIATE = 1,
   AGX3_TEX_OPERAND_REGISTER = 2,
   AGX3_TEX_OPERAND_UNIFORM = 3,
};

/*
 * SMP's f83/f112 pair belongs to the whole sample ABI (descriptor heaps,
 * coordinate staging, derivatives), not to the shader stage. The driver's ABI
 * is f83=0/f112=4; Apple's fragment ABI (f83=4/f112=8) is only for byte-exact
 * witness tests and must be chosen explicitly.
 */
enum agx3_tex_sample_abi {
   AGX3_TEX_SAMPLE_ABI_DRIVER = 0,
   AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT = 1,
};

/*
 * How the level staged into texture operand bank 6 is typed (PACK_SMPIR f74):
 * a filtered sample's level is FLOAT (2), a texel fetch's level INTEGER (3).
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
    * The level: `lod_x64` is an immediate level in 1/64 steps in f29 (0 by
    * default). `register_lod` reads a per-lane level instead (f37 clear,
    * f139=4), published into operand bank 6 by agx3_tex_stage_typed_lod()
    * with AGX3_TEX_LEVEL_INTEGER. AGX3b only.
    */
   uint16_t lod_x64;
   bool register_lod;
   /*
    * Sparse residency: f142=1 requests it and f120=1 marks the access. The
    * residency code is one more 32-bit register after the masked channels
    * (dest + channels * halves), as on AGX2. f120 lies past the 112-bit ep3
    * form, so a sparse operation is always the 20-byte ep4 form. Apple also
    * changes the f112 operand profile, but that profile makes sparse fetches
    * and explicit-LOD sparse samples fail on G17P, so the dense profile is
    * kept. AGX3b only.
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
 * Filtered sample (`texture_sample` -> smp with f78=2) LOD shapes:
 *
 *  - AUTO: implicit LOD, fragment only (quad derivatives). f156 ends the
 *    fragment sample group, not each sample; see sample_group_end.
 *  - LEVEL: immediate level (f37=1, f139=2), `lod_x64` = level * 64.
 *  - REGISTER: per-lane half-float level packed by PACK_SMPIR (f139=4). The
 *    fragment marker is present here too; it does not select derivatives.
 *
 * G17P also has 2D-array (register layer) and cube forms, and 3D
 * immediate-level samples (f76=3 selects the third coordinate). Other
 * dimension/LOD/offset/shadow/gather combinations are refused.
 */
enum agx3_tex_lod {
   AGX3_TEX_LOD_AUTO,
   AGX3_TEX_LOD_LEVEL,
   AGX3_TEX_LOD_REGISTER, /* PACK_SMPIR publishes the per-lane level. */
   /* Implicit LOD plus a constant bias: f139=1, bias * 64 in the 12-bit f29
    * the level form uses (saturating at 2047). A zero bias is the plain
    * implicit form. */
   AGX3_TEX_LOD_BIAS,
   /* Implicit LOD plus a per-lane bias: the REGISTER form with f139=3. */
   AGX3_TEX_LOD_BIAS_REGISTER,
   /* Explicit gradients: f139=5 and no fragment marker f156. The gradient
    * scalars are staged before the sample in the banks below, interleaved
    * (ddx.x, ddy.x, ddx.y, ddy.y[, ddx.z, ddy.z]) as the collect
    * agx_emit_tex() builds for nir_tex_src_ddx. */
   AGX3_TEX_LOD_GRADIENT,
};

/* Staging banks of AGX3_TEX_LOD_GRADIENT, in half-register units. A gradient
 * sample also names coordinate bank 0. */
#define AGX3_TEX_GRADIENT_BANK_DDX_X 8u
#define AGX3_TEX_GRADIENT_BANK_DDY_X 10u
#define AGX3_TEX_GRADIENT_BANK_DDX_Y 12u
#define AGX3_TEX_GRADIENT_BANK_DDY_Y 14u
/* gradient3d continues the interleave. */
#define AGX3_TEX_GRADIENT_BANK_DDX_Z 16u
#define AGX3_TEX_GRADIENT_BANK_DDY_Z 18u

/*
 * textureGather: an implicit sample with f144=1, the component in f145 (R
 * absent, G 1, B 2, A 3) and the f74 mask-group bit cleared. 2D-array and
 * cube gathers use the same selector; runtime 2D offsets use f141=2 with the
 * bank-6 publication below, immediate offsets f43/f51/f141. No explicit LOD.
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
   /* As agx3_tex_fetch_desc::sparse, on every sample shape and width. */
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
 * Coordinate-staging copy before smp: smp reads coordinates a signalling
 * producer published on its wait slot. One 32-bit register per copy (dest ==
 * src when already in place), the smp's slot in f12 and a producer dependency
 * as wait bit f30 + slot - 1.
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
