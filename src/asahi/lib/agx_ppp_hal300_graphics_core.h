/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "agx_chip.h"

/* Triangle state, initially captured on M5/25G76 without PSIZ.
 * Each range is a complete PPP payload with its own VDM pointer. */
#define AGX_PPP_HAL300_GRAPHICS_RECORDS 9u
#define AGX_PPP_HAL300_GRAPHICS_BYTES 172u

struct agx_ppp_hal300_graphics_fields {
   uint64_t fragment_program;
   uint64_t varying_table;
   unsigned cf_binding_count;
   unsigned cull_mode;
   /* Native G18 handleScissorUpdates sets primary ISP control bit 16 when
    * using the separately bound scissor buffer (26A428, 0x21963acb0). */
   bool scissor_enable;
   /* G18 encodeISPPrimitiveType (26A428, 0x2196367b8..0x21963681c)
    * selects primary ISP control bit 17 for a biased triangle. The bias
    * record itself is separately bound by the depth-bias/scissor block. */
   bool depth_bias_enable;
   /* G18 setVisibilityResultMode:offset: (26A428, 0x219660660..0x2196606a8)
    * writes mode 0/2/3 into primary ISP bits 14..15 and the 15-bit query
    * slot into the primary record's last word, bits 17..31. Slot units are
    * uint64_t results; maxVisibilityQueryOffset returns 0x3fff8. The native
    * emitter copies both fields at 0x21963501c..0x2196350a4. Zero defaults
    * preserve the existing non-query path; G17 has no witness here.
    */
   unsigned occlusion_query_mode;
   unsigned occlusion_query_index;
   /*
    * Set when the render has no colour attachment. Two things move with it,
    * both from controlled colour-free renders on 25G76: the interpolated
    * varying count moves from the first byte of the count word to the
    * third, and the ISP record's last word goes from 0x0001ffff to zero.
    * Witnessed with four noperspective varyings and a discarding shader
    * (count word 0x00040000), the exact shape the depth-format conformance
    * tests draw with, and with no varyings at all (count word zero).
    *
    * A flat colour-free varying has no witness and is not admitted here.
    * A colour-free shader that writes depth is carried by
    * fragment_writes_depth below.
    */
   bool no_color_attachment;
   /*
    * ISP pass type: 0 opaque, 1 translucent, 2 punch through, and on G18G
    * 3 translucent punch through. The G18 native constructor selects 3
    * at 0x219564774..0x2195647f0; setFragmentVariant copies all three pass
    * bits at 0x21969a494..0x21969a49c and the secondary ISP emitter preserves
    * them at 0x2196350c0..0x2196350f4 (26A428). Original-instruction replay
    * covers combined tile reads/coverage, primitive and depth-output state.
    * A controlled
    * capture of a discarding fragment shader on 25G76
    * (gpu/m5-state-matrix-reference.m, mode "discard") emitted word1
    * 0x44000200 against the opaque baseline's 0x00000200: the pass type in
    * bits 29..31 and, alongside it, the fragment control's own
    * disable-triangle-merging bit 26. That bit is a property of the shader,
    * not of the pass type, so it is carried separately rather than derived.
    */
   unsigned fragment_pass;
   bool disable_tri_merging;
   /*
    * The fragment shader writes gl_FragDepth, so the depth value is the
    * shader's rather than the rasterizer's. Two things move with it, both
    * from a controlled 25G76 render whose fragment function returns a
    * [[depth(any)]] struct (gpu/m5-state-matrix-reference.m mode
    * "depth-only-zout") against one whose depth comes from interpolation:
    * the fragment program's VDM pipeline template takes 0x40, and both of
    * the ISP record's early-test words drop 0x00c00000, which is the
    * hardware being told it may no longer resolve depth before the shader
    * runs. Writing stencil from the shader is a separate axis with no
    * witness and is not carried here.
    */
   bool fragment_writes_depth;
   bool front_face_ccw;
   bool point_size_output;
   /*
    * Words in the vertex shader's clip-distance UVS group, zero for none.
    * Captured on 25G76 (gpu/m5-state-matrix-reference.m, mode "clipdist")
    * against the identical colour render in the same session: the UVS
    * record's first word gains bit 0 and its word count gains one, where
    * the captured shader exports a single clip distance. Nothing else in
    * the graphics state moves, and the same count appears twice in the VDM
    * prefix. That count is a word total -- position contributes four and
    * each varying component one -- so a wider group adds its own width the
    * same way; only the one-word group is captured.
    */
   unsigned clip_distance_words;
   /*
    * The primitive the rasterizer draws. Triangles are the zero default.
    * Points were captured in isolation on 25G76
    * (gpu/m5-state-matrix-reference.m, mode "points") against the identical
    * colour render in the same session: the depth/stencil record's control
    * word and the ISP record's first word each gain 0x04000000, and both of
    * the ISP record's early-test words gain 0x40000000. Nothing else in the
    * graphics state moves -- the point-size output and the extra vertex
    * component it adds are already carried by point_size_output, and the
    * capture confirms both are set for a point draw.
    *
    * The ISP word's 0x04000000 is the same bit disable_tri_merging sets, so
    * the two are OR'd rather than added: merging triangles is meaningless
    * for points and the hardware is told so either way.
    *
    * Lines have their own oracle mode but no M5 capture yet, so G17G admits
    * only points and triangles. G18G's J873 "lines" render (sessions
    * 20260925-a/b) against its colour render in the same session moves
    * exactly the words points move, except that the early-test words gain
    * 0x10000000 where points set 0x40000000; lines export no point size and
    * keep the colour render's UVS record. Lines are admitted on G18G alone.
    */
   enum agx_ppp_hal300_object_type {
      AGX_PPP_HAL300_TRIANGLES = 0,
      AGX_PPP_HAL300_POINTS = 1,
      AGX_PPP_HAL300_LINES = 2,
   } object_type;
   /* Vertex exports and fragment interpolation are independent. A fragment
    * shader may leave the vertex shader's output unused. Counts are scalar
    * 32-bit UVS slots, not source-language vectors or CF bindings. The
    * interpolator reserves all exported user slots even when a CF binding
    * reads only a subvector; its offset is carried by the VIT separately. */
   unsigned vertex_varying_components;
   unsigned fragment_varying_components;
   /*
    * How the fragment varyings are interpolated. This record's first word
    * separates flat from interpolated only: four smooth components encode
    * 0x00000004 and four flat ones 0x00000400, and a controlled
    * noperspective render encodes 0x00000004 as well, identical to smooth.
    * The perspective-versus-linear distinction is carried per binding by
    * the varying table's shade model, which the linker already writes, so
    * noperspective needs nothing here.
    *
    * (A colour-free render was observed with 0x00040000 in this word. That
    * is not interpolation -- its varying is ordinary smooth -- and it is
    * not understood yet, so no caller may request it.)
    */
   enum agx_ppp_hal300_interpolation {
      AGX_PPP_HAL300_SMOOTH = 0,
      AGX_PPP_HAL300_FLAT = 1,
      AGX_PPP_HAL300_LINEAR = 2,
   } interpolation;
   /*
    * Depth and stencil state, decoded from a controlled M5/25G76 capture
    * matrix (gpu/m5-state-matrix-reference.m, one render per comparison):
    *
    *   face word = (compare << 24) | (depth_write ? 0 : 0x200000) | 0xf00
    *
    * with compare in the order never, less, equal, lequal, greater,
    * notequal, gequal, always -- the same order as PIPE_FUNC_*. The
    * driver's previous no-depth constant 0x07200f00 is exactly
    * "always, writes off" under that rule, and its D32 LESS/write
    * constant 0x01000f00 is "less, writes on", so both existing witnessed
    * values fall out of the decode rather than being special cases.
    *
    * The captured depth-disabled state carries 0x40000. G18's native
    * depth/stencil state constructor additionally requires 0xc0000 whenever
    * stencil is active, even with depth testing enabled. The G17 control
    * retains its separately captured 0x80000 stencil bit. A render with a
    * depth attachment but no depth-stencil state is byte-identical to one
    * with no attachment at all.
    */
   /*
    * Set while a depth-stencil state governs the draw. Cleared -- the
    * zero-initialized default -- encodes the inactive depth unit, which is
    * byte-identical whether or not a depth attachment is bound: a render
    * with a D32 attachment and no depth-stencil state produced exactly the
    * no-attachment record in the capture matrix.
    */
   bool depth_active;
   uint8_t depth_compare;
   bool depth_write;
   bool stencil_active;
   /*
    * References occupy the low byte of the face words. G18's native
    * setStencilReferenceValue (26A428, 0x219660584) and separate-face setter
    * (0x2196605d8) replace exactly those bytes at context +0x7ad4/+0x7ad8.
    * The captured face bit 0 was reference 1, not a write-enable bit.
    * Stencil words retain the common layout: write/read masks at 0/8,
    * depth-pass/depth-fail/stencil-fail operations at 16/19/22 and compare
    * at 25. In particular, captured 0x0e02ffff means always/replace.
    */
   uint8_t front_stencil_reference;
   uint8_t back_stencil_reference;
   uint32_t front_stencil;
   uint32_t back_stencil;
   /*
    * GL_RASTERIZER_DISCARD, as Apple's rasterizationEnabled = NO with a
    * fragment function bound: the W-clamp/cull-2 record's third word gains
    * bit 17 (0x80 -> 0x00020080) and nothing else here moves. 26A428
    * AGXMetalG18G constructRenderPipeline sets pipe+0x3ec bit 17
    * (0x219462740..760) and the draw emits that word unchanged
    * (0x2196353e4..440); the cull record keeps its bit 17 clear. The other
    * half of the state is the VDM vertex-program variant
    * (agx_vdm_state_update_draw_info::rasterization_enabled). See
    * hv/G18-RASTERIZER-DISCARD.md. G18G only: G17G has no witness.
    */
   bool rasterizer_discard;
   /*
    * G18G general varyings (hv/G18-VARYINGS.md): R1, R2 and R4 take the
    * linker's words unchanged. AGXMetalG18G builds R1 from the compiler's
    * per-class counts, AGX2 VARYING_COUNTS for 32- and 16-bit varyings
    * (0x21954bbf8..0x21954bc6c), and R4 word 1 is AGX2 OUTPUT_SELECT, vertex
    * half | fragment half (0x21954b7c8, 0x2195648a0); word 4 is the vertex
    * output total. When set, vertex/fragment_varying_components,
    * interpolation, point_size_output and clip_distance_words are ignored.
    */
   bool general_varyings;
   uint32_t varying_counts_32; /* R1 word 1 */
   uint32_t varying_counts_16; /* R1 word 2 */
   uint32_t output_select;     /* R4 word 1 */
   uint32_t uvs_words;         /* R4 word 4 */
   /* Cull-2 bit 12: the fragment shader reads a hardware primitive ID
    * (Apple: type-7 iteration -> fv+0x1040 -> pipe+0x3ec bit 12,
    * 0x21969a420). */
   bool needs_primitive_id;
   /* Cull word bits 7..8 hold the provoking vertex selector plus one: 0
    * first, 2 last (setProvokingVertexMode 0x2196601f8; the J873 captures
    * carry 0x480 for first and 0x580 for last, gears 0x10582). G18G only. */
   unsigned flat_shading_vertex;
};

/* The two grounded stencil operation encodings. */
#define AGX_PPP_HAL300_STENCIL_KEEP_ALWAYS UINT32_C(0x0e000000)
#define AGX_PPP_HAL300_STENCIL_REPLACE_ALWAYS UINT32_C(0x0e020000)
#define AGX_PPP_HAL300_STENCIL_KEEP_EQUAL UINT32_C(0x04000000)

struct agx_ppp_hal300_graphics_core {
   uint8_t bytes[AGX_PPP_HAL300_GRAPHICS_BYTES];
   uint16_t offsets[AGX_PPP_HAL300_GRAPHICS_RECORDS];
   uint8_t lengths[AGX_PPP_HAL300_GRAPHICS_RECORDS];
};

/* Opaque/translucent triangles, optionally up to two vec4 varyings
 * and D32 LESS/write.
 * Stencil state uses the shared word layout and separate face references.
 * Queries and MSAA need additional state. Attachment storage and clear/load/store
 * setup are separate; packing this state alone does not enable depth rendering.
 * Caller must validate those semantic restrictions before using this profile.
 * On failure out is untouched. */
bool agx_ppp_hal300_graphics_core_pack(
   enum agx_chip chip, const struct agx_ppp_hal300_graphics_fields *fields,
   struct agx_ppp_hal300_graphics_core *out);
