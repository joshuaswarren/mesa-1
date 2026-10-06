/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Display-independent Vulkan graphics proof for T8140/G17P. Render one
 * full-screen triangle into a one-tile image and verify every byte.
 *
 * The default build uses vkCmdCopyImageToBuffer and intentionally exercises
 * Honeykrisp's GPU meta-copy path.  g17p-hostcopy-triangle compiles this same
 * render proof with G17P_HOST_IMAGE_COPY set: it keeps the optimal tiled render
 * target but reads it back on the CPU through VK_EXT_host_image_copy.  Keeping
 * the paths as separate binaries lets render bring-up prove TA/3D/PBE output
 * without hiding regressions in the eventual GPU copy implementation.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef G17P_F16_IO
#define G17P_F16_IO 0
#endif

#ifndef G17P_VERTEX_OUTPUT_68
#define G17P_VERTEX_OUTPUT_68 0
#endif

#ifndef G17P_SAMPLE_RATE_SHADING
#define G17P_SAMPLE_RATE_SHADING 0
#endif

/* 1: dynamic 4x sample mask, 2: dynamic alpha-to-coverage. */
#ifndef G17P_MSAA_COVERAGE
#define G17P_MSAA_COVERAGE 0
#endif

#ifndef G17P_RG16_UNORM
#define G17P_RG16_UNORM 0
#endif

#ifndef G17P_RGBA16_UNORM
#define G17P_RGBA16_UNORM 0
#endif

#ifndef G17P_RGBA16_UINT
#define G17P_RGBA16_UINT 0
#endif

#ifndef G17P_RGBA16_SINT
#define G17P_RGBA16_SINT 0
#endif

#ifndef G17P_RGBA32_UINT
#define G17P_RGBA32_UINT 0
#endif

#ifndef G17P_RGBA32_SINT
#define G17P_RGBA32_SINT 0
#endif

#ifndef G17P_RG16_SNORM
#define G17P_RG16_SNORM 0
#endif

#ifndef G17P_R32_UINT
#define G17P_R32_UINT 0
#endif

#ifndef G17P_R32_SFLOAT
#define G17P_R32_SFLOAT 0
#endif

#ifndef G17P_R32_SINT
#define G17P_R32_SINT 0
#endif

#ifndef G17P_RG16_UINT
#define G17P_RG16_UINT 0
#endif

#ifndef G17P_RG16_SINT
#define G17P_RG16_SINT 0
#endif

#ifndef G17P_RG32_SINT
#define G17P_RG32_SINT 0
#endif

#ifndef G17P_RG32_SFLOAT
#define G17P_RG32_SFLOAT 0
#endif

#ifndef G17P_RG8_UINT
#define G17P_RG8_UINT 0
#endif

#ifndef G17P_RG8_SINT
#define G17P_RG8_SINT 0
#endif

#ifndef G17P_RG32_UINT
#define G17P_RG32_UINT 0
#endif

#ifndef G17P_RGBA8_UINT
#define G17P_RGBA8_UINT 0
#endif

#ifndef G17P_RGBA8_SINT
#define G17P_RGBA8_SINT 0
#endif

#ifndef G17P_RGBA8_SNORM
#define G17P_RGBA8_SNORM 0
#endif

#ifndef G17P_BGRA8_SNORM
#define G17P_BGRA8_SNORM 0
#endif

#ifndef G17P_BGRA8_UNORM
#define G17P_BGRA8_UNORM 0
#endif

#ifndef G17P_BGRA8_SRGB
#define G17P_BGRA8_SRGB 0
#endif

#ifndef G17P_RG8_SNORM
#define G17P_RG8_SNORM 0
#endif

#ifndef G17P_R8_UINT
#define G17P_R8_UINT 0
#endif

#ifndef G17P_R8_UNORM
#define G17P_R8_UNORM 0
#endif

#ifndef G17P_R8_SNORM
#define G17P_R8_SNORM 0
#endif

#ifndef G17P_R8_SINT
#define G17P_R8_SINT 0
#endif

#ifndef G17P_RGBA16_SNORM
#define G17P_RGBA16_SNORM 0
#endif

#ifndef G17P_R16_SNORM
#define G17P_R16_SNORM 0
#endif

#ifndef G17P_R16_UNORM
#define G17P_R16_UNORM 0
#endif

#ifndef G17P_R16_UINT
#define G17P_R16_UINT 0
#endif

#ifndef G17P_R16_SINT
#define G17P_R16_SINT 0
#endif

#ifndef G17P_R16_SFLOAT
#define G17P_R16_SFLOAT 0
#endif

#ifndef G17P_RG16_SFLOAT
#define G17P_RG16_SFLOAT 0
#endif

#ifndef G17P_RGBA16_SFLOAT
#define G17P_RGBA16_SFLOAT 0
#endif

#ifndef G17P_A2B10G10R10_UINT
#define G17P_A2B10G10R10_UINT 0
#endif

#ifndef G17P_A2R10G10B10_UNORM
#define G17P_A2R10G10B10_UNORM 0
#endif

#ifndef G17P_RGBA32_SFLOAT
#define G17P_RGBA32_SFLOAT 0
#endif

/* 1/2 = direct-count indirect nonindexed/indexed, 3/4 = count variants,
 * 5 = indexed indirect with robustBufferAccess enabled. */
#ifndef G17P_INDIRECT_DRAW
#define G17P_INDIRECT_DRAW 0
#endif

/* vkQuake records every one of its drawing contexts into its own secondary
 * command buffer and replays them with vkCmdExecuteCommands(cb, 1, &secondary)
 * called once per secondary, several times in a row, inside a single render
 * pass. The plain secondary oracle executes exactly one secondary, so this
 * variant records four and replays them one call at a time. */
#ifndef G17P_SECONDARY_COMMAND_BUFFER_MULTI
#define G17P_SECONDARY_COMMAND_BUFFER_MULTI 0
#endif

/* The multi variant is the single-secondary variant plus more secondaries, so
 * it reuses every guard the single-secondary oracle already carries. */
#ifndef G17P_SECONDARY_COMMAND_BUFFER
#define G17P_SECONDARY_COMMAND_BUFFER G17P_SECONDARY_COMMAND_BUFFER_MULTI
#endif

#if G17P_SECONDARY_COMMAND_BUFFER_MULTI
#define SECONDARY_QUADRANT_COUNT 4
/* Two triangles per quadrant, selected out of one shared vertex buffer by
 * firstVertex. */
#define SECONDARY_QUADRANT_VERTICES 6
/* Every expected channel is an exact UNORM8 code point, so this absorbs a
 * one-LSB readback quirk and nothing else. */
#define SECONDARY_QUADRANT_TOLERANCE 1
#endif

#if G17P_INDIRECT_DRAW < 0 || G17P_INDIRECT_DRAW > 5
#error "select a valid G17P indirect draw oracle"
#endif

#define G17P_INDIRECT_INDEXED                                                \
   (G17P_INDIRECT_DRAW == 2 || G17P_INDIRECT_DRAW == 4 ||                    \
    G17P_INDIRECT_DRAW == 5)
#define G17P_INDIRECT_COUNT                                                   \
   (G17P_INDIRECT_DRAW == 3 || G17P_INDIRECT_DRAW == 4)
#define G17P_ROBUST_INDEXED (G17P_INDIRECT_DRAW == 5)

#ifndef G17P_INDEX_RESTART
#define G17P_INDEX_RESTART 0
#endif

#ifndef G17P_INSTANCE_RATE
#define G17P_INSTANCE_RATE 0
#endif

#ifndef G17P_VERTEX_FORMAT
#define G17P_VERTEX_FORMAT 0
#endif

#ifndef G17P_IMPLICIT_LOD
#define G17P_IMPLICIT_LOD 0
#endif

#ifndef G17P_TEXTURE_ALPHA_COMPOSITE
#define G17P_TEXTURE_ALPHA_COMPOSITE 0
#endif

#ifndef G17P_COMPOSITE_SPLIT
#define G17P_COMPOSITE_SPLIT 0
#endif

#ifndef G17P_COMPOSITE_FRACTIONAL_INSET
#define G17P_COMPOSITE_FRACTIONAL_INSET 0
#endif

#ifndef G17P_COMPOSITE_NO_FLAT
#define G17P_COMPOSITE_NO_FLAT 0
#endif

#ifndef G17P_COMPOSITE_EXPLICIT_LOD
#define G17P_COMPOSITE_EXPLICIT_LOD 0
#endif

#ifndef G17P_COMPOSITE_SINGLE_DRAW
#define G17P_COMPOSITE_SINGLE_DRAW 0
#endif

#ifndef G17P_COMPOSITE_NARROW_SINGLE
#define G17P_COMPOSITE_NARROW_SINGLE 0
#endif

#ifndef G17P_COMPOSITE_REPEATED_FULL
#define G17P_COMPOSITE_REPEATED_FULL 0
#endif

#ifndef G17P_COMPOSITE_NARROW_TWO
#define G17P_COMPOSITE_NARROW_TWO 0
#endif

#ifndef G17P_COMPOSITE_NARROW_TWO_CHANGED
#define G17P_COMPOSITE_NARROW_TWO_CHANGED 0
#endif

#ifndef G17P_COMPOSITE_NARROW_X
#define G17P_COMPOSITE_NARROW_X 0
#endif

#ifndef G17P_COMPOSITE_NARROW_WIDTH
#define G17P_COMPOSITE_NARROW_WIDTH 2
#endif

#ifndef G17P_COMPOSITE_NARROW_Y
#define G17P_COMPOSITE_NARROW_Y 0
#endif

/* Zero retains the full render-target height. */
#ifndef G17P_COMPOSITE_NARROW_HEIGHT
#define G17P_COMPOSITE_NARROW_HEIGHT 0
#endif

#ifndef G17P_COMPOSITE_FULLSCREEN_TRIANGLE
#define G17P_COMPOSITE_FULLSCREEN_TRIANGLE 0
#endif

#if (G17P_COMPOSITE_SINGLE_DRAW + G17P_COMPOSITE_NARROW_SINGLE +            \
     G17P_COMPOSITE_REPEATED_FULL + G17P_COMPOSITE_NARROW_TWO +             \
     G17P_COMPOSITE_NARROW_TWO_CHANGED) > 1
#error "select one alpha-composite draw/scissor diagnostic"
#endif

#if G17P_COMPOSITE_NO_FLAT && G17P_COMPOSITE_EXPLICIT_LOD
#error "select one alpha-composite diagnostic shader"
#endif

#if G17P_COMPOSITE_FULLSCREEN_TRIANGLE && G17P_COMPOSITE_FRACTIONAL_INSET
#error "the oversized triangle and fractional quad are mutually exclusive"
#endif

#define G17P_COMPOSITE_OPACITY_COUNT 17

#if G17P_COMPOSITE_SPLIT < 0 || G17P_COMPOSITE_SPLIT > 3
#error "G17P_COMPOSITE_SPLIT must select one of two diagonals and windings"
#endif

/* vkQuake's world fragment shape: two distinct sampled textures multiplied
 * together in one invocation. Every other G17P texture oracle, the whole
 * alpha-composite family included, takes exactly one sample per fragment, so
 * a second sampled texture that silently reads zero satisfies all of them
 * while collapsing diffuse * lightmap to black on every BSP surface. Both
 * sources are uniform at every level, so the product is exact whichever mip
 * implicit derivatives select. */
#ifndef G17P_TWO_SAMPLERS
#define G17P_TWO_SAMPLERS 0
#endif

/* Move the second sampler into its own descriptor set. vkQuake binds the
 * diffuse texture at set 0 and the lightmap at set 1, so this variant is the
 * engine's exact binding shape and the one-set variant above is its control:
 * a difference between the two is attributable to the second set alone. */
#ifndef G17P_TWO_SAMPLERS_SPLIT_SETS
#define G17P_TWO_SAMPLERS_SPLIT_SETS 0
#endif

#if G17P_TWO_SAMPLERS_SPLIT_SETS && !G17P_TWO_SAMPLERS
#error "the split-set variant requires the two-sampler oracle"
#endif

/* Texture A is opaque (255,128,64) and texture B is opaque mid-grey. Their
 * UNORM8 product is R 255*128/255 = 128.0, G 128*128/255 = 64.25 and
 * B 64*128/255 = 32.125, with the constant opaque alpha the shader writes.
 * No channel is zero and no two are equal, so a dropped, swapped, or
 * duplicated sampler is all distinguishable; a black world surface — the
 * failure being hunted — sits 128, 64, and 32 code points away. */
#define G17P_TWO_SAMPLERS_SOURCE_A {0xff, 0x80, 0x40, 0xff}
#define G17P_TWO_SAMPLERS_SOURCE_B {0x80, 0x80, 0x80, 0xff}
#define G17P_TWO_SAMPLERS_EXPECTED {128, 64, 32, 255}

/* Both sources are uniform at every level, so filtering and mip selection
 * cannot move the result; one code point of slack per channel absorbs a
 * differently rounded multiply without admitting a missing texture. */
#define G17P_TWO_SAMPLERS_TOLERANCE 2

#ifndef G17P_EXPLICIT_GRADIENT
#define G17P_EXPLICIT_GRADIENT 0
#endif

#ifndef G17P_TEXTURE_GATHER
#define G17P_TEXTURE_GATHER 0
#endif

#ifndef G17P_SATURATE_TEXTURE
#define G17P_SATURATE_TEXTURE 0
#endif

#ifndef G17P_SHADOW_COMPARE
#define G17P_SHADOW_COMPARE 0
#endif

#ifndef G17P_TEXTURE_MIN_LOD
/* 1=tex, 2=txb, 3=txl, 4=txd; image-view clamp is selected at runtime. */
#define G17P_TEXTURE_MIN_LOD 0
#endif
#if G17P_TEXTURE_MIN_LOD
static float view_min_lod;
#endif

#ifndef G17P_TEXEL_FETCH_LOD
#define G17P_TEXEL_FETCH_LOD 0
#endif

#ifndef G17P_CUBE_SAMPLE
#define G17P_CUBE_SAMPLE 0
#endif

#ifndef G17P_TEXTURE_3D
#define G17P_TEXTURE_3D 0
#endif

#ifndef G17P_YCBCR
#define G17P_YCBCR 0
#endif

#ifndef G17P_MULTIVIEW
#define G17P_MULTIVIEW 0
#endif

#ifndef G17P_LAYER_OUTPUT
#define G17P_LAYER_OUTPUT 0
#endif

#ifndef G17P_MULTI_VIEWPORT
#define G17P_MULTI_VIEWPORT 0
#endif

#ifndef G17P_SHADER_QUAD_CONTROL
#define G17P_SHADER_QUAD_CONTROL 0
#endif

/* Exercise native uncompressed sRGB textures and an sRGB attachment. Keep
 * these as modifiers of the existing compositor-style alpha oracle so
 * implicit mip selection, linear filtering, opacity, and premultiplied
 * source-over remain unchanged. GTK's Vulkan renderer can select these source
 * formats directly; Mutter's current Cogl/Zink wallpaper upload uses GL_RGBA8
 * and therefore remains a distinct UNORM path. */
#ifndef G17P_COMPOSITE_SRGB_SOURCE
#define G17P_COMPOSITE_SRGB_SOURCE 0
#endif

#ifndef G17P_COMPOSITE_SRGB_SOURCE_BGRA
#define G17P_COMPOSITE_SRGB_SOURCE_BGRA 0
#endif

#ifndef G17P_COMPOSITE_SRGB_TARGET
#define G17P_COMPOSITE_SRGB_TARGET 0
#endif

#ifndef G17P_COMPOSITE_GENERATE_MIPS
#define G17P_COMPOSITE_GENERATE_MIPS 0
#endif

#ifndef G17P_COMPOSITE_GENERATE_MIPS_ODD
#define G17P_COMPOSITE_GENERATE_MIPS_ODD 0
#endif

/* Descriptor-path discriminator for the same compositor sample:
 * 0 = allocated combined array (original oracle)
 * 1 = allocated single combined image sampler
 * 2 = allocated separate sampled image and sampler
 * Pushed combined/separate cells are exercised by vk_meta's blit path. */
#ifndef G17P_COMPOSITE_DESCRIPTOR_CELL
#define G17P_COMPOSITE_DESCRIPTOR_CELL 0
#endif

#ifndef G17P_COMPOSITE_DESCRIPTOR_ARRAY
#define G17P_COMPOSITE_DESCRIPTOR_ARRAY 0
#endif

#ifndef G17P_COMPOSITE_ARRAY_NEAREST_EXACT
#define G17P_COMPOSITE_ARRAY_NEAREST_EXACT 0
#endif

#if G17P_COMPOSITE_DESCRIPTOR_CELL < 0 || G17P_COMPOSITE_DESCRIPTOR_CELL > 2
#error "G17P_COMPOSITE_DESCRIPTOR_CELL must be 0, 1, or 2"
#endif

#if G17P_COMPOSITE_DESCRIPTOR_CELL && !G17P_TEXTURE_ALPHA_COMPOSITE
#error "descriptor cells require the texture alpha-composite oracle"
#endif

#if G17P_COMPOSITE_DESCRIPTOR_ARRAY && !G17P_COMPOSITE_DESCRIPTOR_CELL
#error "descriptor array view requires an allocated descriptor cell"
#endif

#if G17P_COMPOSITE_ARRAY_NEAREST_EXACT && !G17P_COMPOSITE_DESCRIPTOR_ARRAY
#error "nearest array sample requires an allocated array descriptor cell"
#endif

#if (G17P_COMPOSITE_SRGB_SOURCE || G17P_COMPOSITE_SRGB_TARGET) && \
   !G17P_TEXTURE_ALPHA_COMPOSITE
#error "sRGB composite modifiers require the texture alpha-composite oracle"
#endif

#if G17P_COMPOSITE_SRGB_SOURCE_BGRA && !G17P_COMPOSITE_SRGB_SOURCE
#error "the BGRA source modifier requires an sRGB composite source"
#endif

#if G17P_COMPOSITE_SRGB_TARGET && !G17P_COMPOSITE_SRGB_SOURCE
#error "the sRGB target oracle requires an sRGB composite source"
#endif

#if G17P_COMPOSITE_GENERATE_MIPS && !G17P_TEXTURE_ALPHA_COMPOSITE
#error "generated mipmaps require the texture alpha-composite oracle"
#endif

#if G17P_COMPOSITE_GENERATE_MIPS_ODD && !G17P_COMPOSITE_GENERATE_MIPS
#error "odd generated mipmaps require generated mipmaps"
#endif

#if G17P_COMPOSITE_SRGB_SOURCE || G17P_COMPOSITE_SRGB_TARGET
#include "util/format_srgb.h"
#endif

#if G17P_MULTI_VIEWPORT < 0 || G17P_MULTI_VIEWPORT > 4
#error "select a valid static/dynamic multi-viewport oracle"
#endif

#define G17P_MULTI_VIEWPORT_DYNAMIC_VIEWPORT                              \
   (G17P_MULTI_VIEWPORT == 2 || G17P_MULTI_VIEWPORT == 4)
#define G17P_MULTI_VIEWPORT_DYNAMIC_SCISSOR                               \
   (G17P_MULTI_VIEWPORT == 3 || G17P_MULTI_VIEWPORT == 4)

#define G17P_LOD_ORACLE (G17P_IMPLICIT_LOD || G17P_EXPLICIT_GRADIENT)
#define G17P_TEXTURE_ORACLE                                                \
   (G17P_LOD_ORACLE || G17P_TEXTURE_ALPHA_COMPOSITE ||                     \
    G17P_TEXTURE_GATHER || G17P_SATURATE_TEXTURE || G17P_SHADOW_COMPARE || \
    G17P_TEXEL_FETCH_LOD || G17P_CUBE_SAMPLE || G17P_TEXTURE_3D ||          \
    G17P_TWO_SAMPLERS || G17P_YCBCR || G17P_TEXTURE_MIN_LOD)

#if G17P_YCBCR
#define TEXTURE_FORMAT VK_FORMAT_G8_B8R8_2PLANE_420_UNORM
#define TEXTURE_ASPECT VK_IMAGE_ASPECT_COLOR_BIT
#define TEXTURE_PIXEL_BYTES 1
#elif G17P_SHADOW_COMPARE
#define TEXTURE_FORMAT VK_FORMAT_D16_UNORM
#define TEXTURE_ASPECT VK_IMAGE_ASPECT_DEPTH_BIT
#define TEXTURE_PIXEL_BYTES 2
#elif G17P_COMPOSITE_SRGB_SOURCE_BGRA
#define TEXTURE_FORMAT VK_FORMAT_B8G8R8A8_SRGB
#define TEXTURE_ASPECT VK_IMAGE_ASPECT_COLOR_BIT
#define TEXTURE_PIXEL_BYTES 4
#elif G17P_COMPOSITE_GENERATE_MIPS
#define TEXTURE_FORMAT VK_FORMAT_B8G8R8A8_UNORM
#define TEXTURE_ASPECT VK_IMAGE_ASPECT_COLOR_BIT
#define TEXTURE_PIXEL_BYTES 4
#elif G17P_COMPOSITE_SRGB_SOURCE
#define TEXTURE_FORMAT VK_FORMAT_R8G8B8A8_SRGB
#define TEXTURE_ASPECT VK_IMAGE_ASPECT_COLOR_BIT
#define TEXTURE_PIXEL_BYTES 4
#else
#define TEXTURE_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define TEXTURE_ASPECT VK_IMAGE_ASPECT_COLOR_BIT
#define TEXTURE_PIXEL_BYTES 4
#endif

#ifndef G17P_DYNAMIC_VERTEX_INPUT
#define G17P_DYNAMIC_VERTEX_INPUT 0
#endif

#if G17P_VERTEX_FORMAT < 0 || G17P_VERTEX_FORMAT > 3
#error "select a valid G17P vertex-format oracle"
#endif

/* The dynamic variant reuses a vertex-format oracle's data and expected image
 * and changes only how the vertex input is supplied, so that a difference is
 * attributable to VK_EXT_vertex_input_dynamic_state alone. */
#if G17P_DYNAMIC_VERTEX_INPUT && !G17P_VERTEX_FORMAT
#error "dynamic vertex input needs a vertex-format oracle to drive"
#endif

#if G17P_VERTEX_FORMAT &&                                                  \
   (G17P_INSTANCE_RATE || G17P_INDEX_RESTART || G17P_INDIRECT_DRAW)
#error "vertex-format oracles cannot be combined with another draw oracle"
#endif

#ifndef G17P_BRESENHAM_LINES
#define G17P_BRESENHAM_LINES 0
#endif

#ifndef G17P_LOGIC_OP_XOR
#define G17P_LOGIC_OP_XOR 0
#endif

#ifndef G17P_BLEND_CONSTANT
#define G17P_BLEND_CONSTANT 0
#endif

#ifndef G17P_RGBA8_MRT
#define G17P_RGBA8_MRT 0
#endif

#ifndef G17P_STK_R11G11B10_MRT
#define G17P_STK_R11G11B10_MRT 0
#endif

#ifndef G17P_DYNAMIC_BLEND
#define G17P_DYNAMIC_BLEND 0
#endif

#ifndef G17P_ADVANCED_BLEND
#define G17P_ADVANCED_BLEND 0
#endif

#ifndef G17P_RASTER_DISCARD
#define G17P_RASTER_DISCARD 0
#endif

#if (G17P_F16_IO + G17P_VERTEX_OUTPUT_68 + G17P_RG16_UNORM +           \
     G17P_RGBA16_UNORM +                                                \
     G17P_RGBA16_UINT + G17P_RGBA16_SINT + G17P_RGBA32_UINT +           \
     G17P_RGBA32_SINT +                                                  \
     G17P_RG16_SNORM + G17P_R32_UINT + G17P_R32_SFLOAT +              \
     G17P_R32_SINT + G17P_RG16_UINT + G17P_RG16_SINT +                \
     G17P_RG32_SINT + G17P_RG32_SFLOAT + G17P_RG8_UINT +              \
     G17P_RG8_SINT + G17P_RG32_UINT + G17P_RGBA8_UINT +               \
     G17P_RGBA8_SINT + G17P_RGBA8_SNORM + G17P_BGRA8_SNORM +          \
     G17P_BGRA8_UNORM + G17P_BGRA8_SRGB +                              \
     G17P_RG8_SNORM +                                                   \
     G17P_R8_UINT + G17P_R8_UNORM + G17P_R8_SNORM + G17P_R8_SINT +    \
     G17P_RGBA16_SNORM +                                               \
     G17P_R16_SNORM + G17P_R16_UNORM + G17P_R16_UINT +               \
     G17P_R16_SINT + G17P_R16_SFLOAT + G17P_RG16_SFLOAT +             \
     G17P_RGBA16_SFLOAT + G17P_A2B10G10R10_UINT +                    \
     G17P_A2R10G10B10_UNORM + G17P_RGBA32_SFLOAT +                    \
     (G17P_VERTEX_FORMAT != 0) + G17P_BRESENHAM_LINES +                   \
     G17P_LOGIC_OP_XOR + G17P_BLEND_CONSTANT + G17P_RGBA8_MRT +           \
     G17P_DYNAMIC_BLEND + G17P_ADVANCED_BLEND + G17P_RASTER_DISCARD +     \
     G17P_SAMPLE_RATE_SHADING + (G17P_MSAA_COVERAGE != 0) +                \
     G17P_TEXTURE_ALPHA_COMPOSITE + G17P_TWO_SAMPLERS + G17P_YCBCR +        \
     G17P_SHADER_QUAD_CONTROL) > 1
#error "select exactly one guarded G17P oracle"
#endif

#if G17P_VERTEX_OUTPUT_68
#include "g17p_vulkan_vertex_output_68_frag.h"
#include "g17p_vulkan_vertex_output_68_vert.h"
#define G17P_VERT_SPV g17p_vulkan_vertex_output_68_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_vertex_output_68_frag_spv
#elif G17P_SAMPLE_RATE_SHADING || G17P_MSAA_COVERAGE == 1
#include "g17p_vulkan_sample_rate_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_sample_rate_frag_spv
#elif G17P_MSAA_COVERAGE == 2
#include "g17p_vulkan_alpha_to_coverage_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_alpha_to_coverage_frag_spv
#elif G17P_MSAA_COVERAGE == 3
/* Deliberately the ordinary single-invocation fragment shader. Variant 1
 * qualifies the dynamic sample mask with the per-sample shader, which also
 * requires sampleRateShading; this variant isolates the mask itself so it can
 * be qualified on a device with no per-sample fragment execution. */
#include "g17p_wsi_triangle_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_wsi_triangle_frag_spv
#elif G17P_VERTEX_FORMAT == 1
#include "g17p_vulkan_vertex_unorm_vert.h"
#include "g17p_wsi_triangle_frag.h"
#define G17P_VERT_SPV g17p_vulkan_vertex_unorm_vert_spv
#define G17P_FRAG_SPV g17p_wsi_triangle_frag_spv
#define VERTEX_INPUT_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#elif G17P_VERTEX_FORMAT == 2
#include "g17p_vulkan_vertex_sint_vert.h"
#include "g17p_wsi_triangle_frag.h"
#define G17P_VERT_SPV g17p_vulkan_vertex_sint_vert_spv
#define G17P_FRAG_SPV g17p_wsi_triangle_frag_spv
#define VERTEX_INPUT_FORMAT VK_FORMAT_R16G16_SINT
#elif G17P_VERTEX_FORMAT == 3
#include "g17p_wsi_triangle_vert.h"
#include "g17p_wsi_triangle_frag.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_wsi_triangle_frag_spv
#define VERTEX_INPUT_FORMAT VK_FORMAT_R16G16_SFLOAT
#elif G17P_RGBA8_MRT
#include "g17p_wsi_triangle_vert.h"
#include "g17p_vulkan_rgba8_mrt_frag.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba8_mrt_frag_spv
#elif G17P_TEXTURE_ALPHA_COMPOSITE
#if G17P_COMPOSITE_NO_FLAT
#include "g17p_vulkan_implicit_lod_vert.h"
#include "g17p_vulkan_texture_alpha_composite_no_flat_frag.h"
#define G17P_VERT_SPV g17p_vulkan_implicit_lod_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_texture_alpha_composite_no_flat_frag_spv
#elif G17P_COMPOSITE_EXPLICIT_LOD
#include "g17p_vulkan_texture_alpha_composite_vert.h"
#include "g17p_vulkan_texture_alpha_composite_lod1_frag.h"
#define G17P_VERT_SPV g17p_vulkan_texture_alpha_composite_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_texture_alpha_composite_lod1_frag_spv
#elif G17P_COMPOSITE_DESCRIPTOR_CELL == 1
#include "g17p_vulkan_texture_alpha_composite_vert.h"
#if G17P_COMPOSITE_DESCRIPTOR_ARRAY
#if G17P_COMPOSITE_ARRAY_NEAREST_EXACT
#include "g17p_vulkan_texture_alpha_composite_single_array_exact_frag.h"
#define G17P_FRAG_SPV \
   g17p_vulkan_texture_alpha_composite_single_array_exact_frag_spv
#else
#include "g17p_vulkan_texture_alpha_composite_single_array_frag.h"
#define G17P_FRAG_SPV g17p_vulkan_texture_alpha_composite_single_array_frag_spv
#endif
#else
#include "g17p_vulkan_texture_alpha_composite_single_frag.h"
#define G17P_FRAG_SPV g17p_vulkan_texture_alpha_composite_single_frag_spv
#endif
#define G17P_VERT_SPV g17p_vulkan_texture_alpha_composite_vert_spv
#elif G17P_COMPOSITE_DESCRIPTOR_CELL == 2
#include "g17p_vulkan_texture_alpha_composite_vert.h"
#if G17P_COMPOSITE_DESCRIPTOR_ARRAY
#if G17P_COMPOSITE_ARRAY_NEAREST_EXACT
#include "g17p_vulkan_texture_alpha_composite_separate_array_exact_frag.h"
#define G17P_FRAG_SPV \
   g17p_vulkan_texture_alpha_composite_separate_array_exact_frag_spv
#else
#include "g17p_vulkan_texture_alpha_composite_separate_array_frag.h"
#define G17P_FRAG_SPV g17p_vulkan_texture_alpha_composite_separate_array_frag_spv
#endif
#else
#include "g17p_vulkan_texture_alpha_composite_separate_frag.h"
#define G17P_FRAG_SPV g17p_vulkan_texture_alpha_composite_separate_frag_spv
#endif
#define G17P_VERT_SPV g17p_vulkan_texture_alpha_composite_vert_spv
#else
#include "g17p_vulkan_texture_alpha_composite_vert.h"
#include "g17p_vulkan_texture_alpha_composite_frag.h"
#define G17P_VERT_SPV g17p_vulkan_texture_alpha_composite_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_texture_alpha_composite_frag_spv
#endif
#elif G17P_TWO_SAMPLERS
/* The plain implicit-LOD vertex shader: one position attribute and one
 * smooth UV, so the only thing these two fragment programs change relative
 * to every other texture oracle is the number of sampled textures. */
#include "g17p_vulkan_implicit_lod_vert.h"
#if G17P_TWO_SAMPLERS_SPLIT_SETS
#include "g17p_vulkan_two_samplers_split_frag.h"
#define G17P_FRAG_SPV g17p_vulkan_two_samplers_split_frag_spv
#else
#include "g17p_vulkan_two_samplers_frag.h"
#define G17P_FRAG_SPV g17p_vulkan_two_samplers_frag_spv
#endif
#define G17P_VERT_SPV g17p_vulkan_implicit_lod_vert_spv
#elif G17P_IMPLICIT_LOD
#include "g17p_vulkan_implicit_lod_frag.h"
#include "g17p_vulkan_implicit_lod_vert.h"
#define G17P_VERT_SPV g17p_vulkan_implicit_lod_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_implicit_lod_frag_spv
#elif G17P_EXPLICIT_GRADIENT
#include "g17p_vulkan_implicit_lod_vert.h"
#include "g17p_vulkan_texture_grad_frag.h"
#define G17P_VERT_SPV g17p_vulkan_implicit_lod_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_texture_grad_frag_spv
#elif G17P_TEXTURE_GATHER
#include "g17p_vulkan_implicit_lod_vert.h"
#include "g17p_vulkan_texture_gather_frag.h"
#define G17P_VERT_SPV g17p_vulkan_implicit_lod_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_texture_gather_frag_spv
#elif G17P_SATURATE_TEXTURE
#include "g17p_vulkan_implicit_lod_vert.h"
#include "g17p_vulkan_saturate_texture_frag.h"
#define G17P_VERT_SPV g17p_vulkan_implicit_lod_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_saturate_texture_frag_spv
#elif G17P_SHADOW_COMPARE
#include "g17p_vulkan_implicit_lod_vert.h"
#include "g17p_vulkan_shadow_compare_frag.h"
#define G17P_VERT_SPV g17p_vulkan_implicit_lod_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_shadow_compare_frag_spv
#elif G17P_TEXTURE_MIN_LOD
#include "g17p_vulkan_implicit_lod_vert.h"
#if G17P_TEXTURE_MIN_LOD == 1
#include "g17p_vulkan_min_lod_tex_frag.h"
#define G17P_FRAG_SPV g17p_vulkan_min_lod_tex_frag_spv
#elif G17P_TEXTURE_MIN_LOD == 2
#include "g17p_vulkan_min_lod_txb_frag.h"
#define G17P_FRAG_SPV g17p_vulkan_min_lod_txb_frag_spv
#elif G17P_TEXTURE_MIN_LOD == 3
#include "g17p_vulkan_min_lod_txl_frag.h"
#define G17P_FRAG_SPV g17p_vulkan_min_lod_txl_frag_spv
#elif G17P_TEXTURE_MIN_LOD == 5
#include "g17p_vulkan_min_lod_txf0_frag.h"
#define G17P_FRAG_SPV g17p_vulkan_min_lod_txf0_frag_spv
#elif G17P_TEXTURE_MIN_LOD == 6
#include "g17p_vulkan_min_lod_txf1_frag.h"
#define G17P_FRAG_SPV g17p_vulkan_min_lod_txf1_frag_spv
#else
#include "g17p_vulkan_min_lod_txd_frag.h"
#define G17P_FRAG_SPV g17p_vulkan_min_lod_txd_frag_spv
#endif
#define G17P_VERT_SPV g17p_vulkan_implicit_lod_vert_spv
#elif G17P_TEXEL_FETCH_LOD
#include "g17p_vulkan_implicit_lod_vert.h"
#include "g17p_vulkan_texel_fetch_lod_frag.h"
#define G17P_VERT_SPV g17p_vulkan_implicit_lod_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_texel_fetch_lod_frag_spv
#elif G17P_CUBE_SAMPLE
#include "g17p_vulkan_implicit_lod_vert.h"
#include "g17p_vulkan_cube_sample_frag.h"
#define G17P_VERT_SPV g17p_vulkan_implicit_lod_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_cube_sample_frag_spv
#elif G17P_TEXTURE_3D
#include "g17p_vulkan_implicit_lod_vert.h"
#include "g17p_vulkan_texture_3d_frag.h"
#define G17P_VERT_SPV g17p_vulkan_implicit_lod_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_texture_3d_frag_spv
#elif G17P_YCBCR
#include "g17p_vulkan_implicit_lod_frag.h"
#include "g17p_vulkan_implicit_lod_vert.h"
#define G17P_VERT_SPV g17p_vulkan_implicit_lod_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_implicit_lod_frag_spv
#elif G17P_INSTANCE_RATE
#include "g17p_vulkan_instance_rate_vert.h"
#include "g17p_wsi_triangle_frag.h"
#define G17P_VERT_SPV g17p_vulkan_instance_rate_vert_spv
#define G17P_FRAG_SPV g17p_wsi_triangle_frag_spv
#elif G17P_INDIRECT_DRAW
#include "g17p_vulkan_indirect_draw_vert.h"
#include "g17p_wsi_triangle_frag.h"
#define G17P_VERT_SPV g17p_vulkan_indirect_draw_vert_spv
#define G17P_FRAG_SPV g17p_wsi_triangle_frag_spv
#elif G17P_A2R10G10B10_UNORM
#include "g17p_vulkan_a2r10g10b10_unorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_a2r10g10b10_unorm_frag_spv
#elif G17P_RGBA32_SFLOAT
#include "g17p_vulkan_rgba32_sfloat_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba32_sfloat_frag_spv
#elif G17P_A2B10G10R10_UINT
#include "g17p_vulkan_a2b10g10r10_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_a2b10g10r10_uint_frag_spv
#elif G17P_RGBA16_SFLOAT
#include "g17p_vulkan_rgba16_sfloat_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba16_sfloat_frag_spv
#elif G17P_RG16_SFLOAT
#include "g17p_vulkan_rg16_sfloat_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg16_sfloat_frag_spv
#elif G17P_R16_SFLOAT
#include "g17p_vulkan_r16_sfloat_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r16_sfloat_frag_spv
#elif G17P_R16_SINT
#include "g17p_vulkan_r16_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r16_sint_frag_spv
#elif G17P_R16_UINT
#include "g17p_vulkan_r16_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r16_uint_frag_spv
#elif G17P_RGBA32_SINT
#include "g17p_vulkan_rgba32_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba32_sint_frag_spv
#elif G17P_RGBA32_UINT
#include "g17p_vulkan_rgba32_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba32_uint_frag_spv
#elif G17P_RGBA16_SINT
#include "g17p_vulkan_rgba16_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba16_sint_frag_spv
#elif G17P_RGBA16_UINT
#include "g17p_vulkan_rgba16_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba16_uint_frag_spv
#elif G17P_R16_UNORM
#include "g17p_vulkan_r16_unorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r16_unorm_frag_spv
#elif G17P_R16_SNORM
#include "g17p_vulkan_r16_snorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r16_snorm_frag_spv
#elif G17P_RGBA16_SNORM
#include "g17p_vulkan_rgba16_snorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba16_snorm_frag_spv
#elif G17P_R8_SINT
#include "g17p_vulkan_r8_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r8_sint_frag_spv
#elif G17P_R8_UNORM
#include "g17p_vulkan_r8_unorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r8_unorm_frag_spv
#elif G17P_R8_SNORM
#include "g17p_vulkan_r8_snorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r8_snorm_frag_spv
#elif G17P_R8_UINT
#include "g17p_vulkan_r8_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r8_uint_frag_spv
#elif G17P_RG8_SNORM
#include "g17p_vulkan_rg8_snorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg8_snorm_frag_spv
#elif G17P_BGRA8_SNORM
#include "g17p_vulkan_rgba8_snorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba8_snorm_frag_spv
#elif G17P_RGBA8_SNORM
#include "g17p_vulkan_rgba8_snorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba8_snorm_frag_spv
#elif G17P_RGBA8_SINT
#include "g17p_vulkan_rgba8_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba8_sint_frag_spv
#elif G17P_RGBA8_UINT
#include "g17p_vulkan_rgba8_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba8_uint_frag_spv
#elif G17P_RG8_SINT
#include "g17p_vulkan_rg8_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg8_sint_frag_spv
#elif G17P_RG8_UINT
#include "g17p_vulkan_rg8_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg8_uint_frag_spv
#elif G17P_RG32_UINT
#include "g17p_vulkan_rg32_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg32_uint_frag_spv
#elif G17P_R32_UINT
#include "g17p_vulkan_r32_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r32_uint_frag_spv
#elif G17P_R32_SINT
#include "g17p_vulkan_r32_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r32_sint_frag_spv
#elif G17P_RG32_SINT
#include "g17p_vulkan_rg32_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg32_sint_frag_spv
#elif G17P_RG32_SFLOAT
#include "g17p_vulkan_rg32_sfloat_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg32_sfloat_frag_spv
#elif G17P_RG16_UINT
#include "g17p_vulkan_rg16_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg16_uint_frag_spv
#elif G17P_RG16_SINT
#include "g17p_vulkan_rg16_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg16_sint_frag_spv
#elif G17P_MULTI_VIEWPORT
#include "g17p_vulkan_viewport_index_vert.h"
#include "g17p_wsi_triangle_frag.h"
#define G17P_VERT_SPV g17p_vulkan_viewport_index_vert_spv
#define G17P_FRAG_SPV g17p_wsi_triangle_frag_spv
#elif G17P_SHADER_QUAD_CONTROL
#include "g17p_vulkan_shader_quad_control_frag.h"
#include "g17p_vulkan_implicit_lod_vert.h"
#define G17P_VERT_SPV g17p_vulkan_implicit_lod_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_shader_quad_control_frag_spv
#elif G17P_LAYER_OUTPUT
#include "g17p_vulkan_layer_vert.h"
#include "g17p_wsi_triangle_frag.h"
#define G17P_VERT_SPV g17p_vulkan_layer_vert_spv
#define G17P_FRAG_SPV g17p_wsi_triangle_frag_spv
#elif G17P_F16_IO
#include "g17p_vulkan_f16_io_frag.h"
#include "g17p_vulkan_f16_io_vert.h"
#define G17P_VERT_SPV g17p_vulkan_f16_io_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_f16_io_frag_spv
#elif G17P_R32_SFLOAT
#include "g17p_vulkan_r32_sfloat_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r32_sfloat_frag_spv
#elif G17P_RG16_SNORM
#include "g17p_vulkan_rg16_snorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg16_snorm_frag_spv
#elif G17P_SECONDARY_COMMAND_BUFFER_MULTI
/* The ordinary position-only vertex shader. The fragment program differs from
 * the constant-colour one only in taking its colour from a push constant, so
 * each secondary can paint its quadrant a distinguishable colour. */
#include "g17p_vulkan_secondary_quadrant_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_secondary_quadrant_frag_spv
#else
#include "g17p_wsi_triangle_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_wsi_triangle_frag_spv
#endif

#define T8140_DEVICE_ID 0x8140
#ifndef IMAGE_WIDTH
#define IMAGE_WIDTH 32
#endif
#ifndef IMAGE_HEIGHT
#define IMAGE_HEIGHT 32
#endif
#if G17P_TEXTURE_ALPHA_COMPOSITE && \
   (IMAGE_WIDTH < 17 || IMAGE_HEIGHT < 1)
#error "the alpha-composite extent must fit all 17 opacity bands"
#endif
#define IMAGE_LAYERS ((G17P_MULTIVIEW || G17P_LAYER_OUTPUT) ? 2 : 1)

#if G17P_STK_R11G11B10_MRT
#define IMAGE_FORMAT VK_FORMAT_B10G11R11_UFLOAT_PACK32
#define IMAGE_PIXEL_BYTES 4
#elif G17P_TEXTURE_ALPHA_COMPOSITE && G17P_COMPOSITE_SRGB_TARGET
#define IMAGE_FORMAT VK_FORMAT_B8G8R8A8_SRGB
#define IMAGE_PIXEL_BYTES 4
#elif G17P_INDIRECT_DRAW
#define IMAGE_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define IMAGE_PIXEL_BYTES 4
#elif G17P_A2R10G10B10_UNORM
#define IMAGE_FORMAT VK_FORMAT_A2R10G10B10_UNORM_PACK32
#define IMAGE_PIXEL_BYTES 4
#define READBACK_GUARD_BYTES 64
#elif G17P_RGBA32_SFLOAT
#define IMAGE_FORMAT VK_FORMAT_R32G32B32A32_SFLOAT
#define IMAGE_PIXEL_BYTES 16
#elif G17P_A2B10G10R10_UINT
#define IMAGE_FORMAT VK_FORMAT_A2B10G10R10_UINT_PACK32
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RGBA16_SFLOAT
#define IMAGE_FORMAT VK_FORMAT_R16G16B16A16_SFLOAT
#define IMAGE_PIXEL_BYTES 8
#elif G17P_RG16_SFLOAT
#define IMAGE_FORMAT VK_FORMAT_R16G16_SFLOAT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_R16_SFLOAT
#define IMAGE_FORMAT VK_FORMAT_R16_SFLOAT
#define IMAGE_PIXEL_BYTES 2
#elif G17P_R16_SINT
#define IMAGE_FORMAT VK_FORMAT_R16_SINT
#define IMAGE_PIXEL_BYTES 2
#elif G17P_R16_UINT
#define IMAGE_FORMAT VK_FORMAT_R16_UINT
#define IMAGE_PIXEL_BYTES 2
#elif G17P_RGBA32_SINT
#define IMAGE_FORMAT VK_FORMAT_R32G32B32A32_SINT
#define IMAGE_PIXEL_BYTES 16
#elif G17P_RGBA32_UINT
#define IMAGE_FORMAT VK_FORMAT_R32G32B32A32_UINT
#define IMAGE_PIXEL_BYTES 16
#elif G17P_RGBA16_SINT
#define IMAGE_FORMAT VK_FORMAT_R16G16B16A16_SINT
#define IMAGE_PIXEL_BYTES 8
#elif G17P_RGBA16_UINT
#define IMAGE_FORMAT VK_FORMAT_R16G16B16A16_UINT
#define IMAGE_PIXEL_BYTES 8
#elif G17P_R16_UNORM
#define IMAGE_FORMAT VK_FORMAT_R16_UNORM
#define IMAGE_PIXEL_BYTES 2
#elif G17P_R16_SNORM
#define IMAGE_FORMAT VK_FORMAT_R16_SNORM
#define IMAGE_PIXEL_BYTES 2
#elif G17P_RGBA16_SNORM
#define IMAGE_FORMAT VK_FORMAT_R16G16B16A16_SNORM
#define IMAGE_PIXEL_BYTES 8
#elif G17P_R8_SINT
#define IMAGE_FORMAT VK_FORMAT_R8_SINT
#define IMAGE_PIXEL_BYTES 1
#elif G17P_R8_UNORM
#define IMAGE_FORMAT VK_FORMAT_R8_UNORM
#define IMAGE_PIXEL_BYTES 1
#define READBACK_GUARD_BYTES 64
#elif G17P_R8_SNORM
#define IMAGE_FORMAT VK_FORMAT_R8_SNORM
#define IMAGE_PIXEL_BYTES 1
#define READBACK_GUARD_BYTES 64
#elif G17P_R8_UINT
#define IMAGE_FORMAT VK_FORMAT_R8_UINT
#define IMAGE_PIXEL_BYTES 1
#elif G17P_RG8_SNORM
#define IMAGE_FORMAT VK_FORMAT_R8G8_SNORM
#define IMAGE_PIXEL_BYTES 2
#define READBACK_GUARD_BYTES 64
#elif G17P_BGRA8_UNORM
#define IMAGE_FORMAT VK_FORMAT_B8G8R8A8_UNORM
#define IMAGE_PIXEL_BYTES 4
#define READBACK_GUARD_BYTES 64
#elif G17P_BGRA8_SRGB
#define IMAGE_FORMAT VK_FORMAT_B8G8R8A8_SRGB
#define IMAGE_PIXEL_BYTES 4
#define READBACK_GUARD_BYTES 64
#elif G17P_BGRA8_SNORM
#define IMAGE_FORMAT VK_FORMAT_B8G8R8A8_SNORM
#define IMAGE_PIXEL_BYTES 4
#define READBACK_GUARD_BYTES 64
#elif G17P_RGBA8_SNORM
#define IMAGE_FORMAT VK_FORMAT_R8G8B8A8_SNORM
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RGBA8_SINT
#define IMAGE_FORMAT VK_FORMAT_R8G8B8A8_SINT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RGBA8_UINT
#define IMAGE_FORMAT VK_FORMAT_R8G8B8A8_UINT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RG8_SINT
#define IMAGE_FORMAT VK_FORMAT_R8G8_SINT
#define IMAGE_PIXEL_BYTES 2
#elif G17P_RG8_UINT
#define IMAGE_FORMAT VK_FORMAT_R8G8_UINT
#define IMAGE_PIXEL_BYTES 2
#elif G17P_RGBA16_UNORM
#define IMAGE_FORMAT VK_FORMAT_R16G16B16A16_UNORM
#define IMAGE_PIXEL_BYTES 8
#elif G17P_RG16_SNORM
#define IMAGE_FORMAT VK_FORMAT_R16G16_SNORM
#define IMAGE_PIXEL_BYTES 4
#elif G17P_R32_SINT
#define IMAGE_FORMAT VK_FORMAT_R32_SINT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RG32_SINT
#define IMAGE_FORMAT VK_FORMAT_R32G32_SINT
#define IMAGE_PIXEL_BYTES 8
#elif G17P_RG32_SFLOAT
#define IMAGE_FORMAT VK_FORMAT_R32G32_SFLOAT
#define IMAGE_PIXEL_BYTES 8
#elif G17P_R32_UINT
#define IMAGE_FORMAT VK_FORMAT_R32_UINT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RG32_UINT
#define IMAGE_FORMAT VK_FORMAT_R32G32_UINT
#define IMAGE_PIXEL_BYTES 8
#elif G17P_R32_SFLOAT
#define IMAGE_FORMAT VK_FORMAT_R32_SFLOAT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RG16_UINT
#define IMAGE_FORMAT VK_FORMAT_R16G16_UINT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RG16_SINT
#define IMAGE_FORMAT VK_FORMAT_R16G16_SINT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RG16_UNORM
#define IMAGE_FORMAT VK_FORMAT_R16G16_UNORM
#define IMAGE_PIXEL_BYTES 4
#else
#define IMAGE_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define IMAGE_PIXEL_BYTES 4
#endif

#define IMAGE_BYTES (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_PIXEL_BYTES)
#define READBACK_PAYLOAD_BYTES (IMAGE_BYTES * IMAGE_LAYERS)

#ifndef G17P_HOST_IMAGE_COPY
#define G17P_HOST_IMAGE_COPY 0
#endif

#ifndef G17P_MSAA_SAMPLES
#define G17P_MSAA_SAMPLES 1
#endif

#if G17P_SAMPLE_RATE_SHADING && G17P_MSAA_SAMPLES != 4
#error "the sample-rate shading oracle requires exactly four samples"
#endif

#if G17P_MSAA_COVERAGE && G17P_MSAA_SAMPLES != 4
#error "the dynamic coverage oracles require exactly four samples"
#endif

#if G17P_MSAA_SAMPLES != 1 && G17P_MSAA_SAMPLES != 2 && \
    G17P_MSAA_SAMPLES != 4
#error "the G17P qualification fixture covers only 1x, 2x and 4x"
#endif

#if G17P_HOST_IMAGE_COPY && G17P_MSAA_SAMPLES != 1
#error "the MSAA fixture validates the GPU resolve and copy path"
#endif

#if G17P_F16_IO && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the f16 I/O fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG16_UNORM && (G17P_F16_IO || G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG16 fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RGBA16_UNORM &&                                               \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RG16_SNORM ||              \
    G17P_R32_UINT || G17P_R32_SINT || G17P_RG32_SFLOAT ||             \
    G17P_HOST_IMAGE_COPY ||                                             \
    G17P_MSAA_SAMPLES != 1)
#error "the RGBA16 fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG16_SNORM &&                                                   \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_R32_UINT || G17P_R32_SINT || G17P_RG32_SFLOAT ||              \
    G17P_HOST_IMAGE_COPY ||                                              \
    G17P_MSAA_SAMPLES != 1)
#error "the RG16 SNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R32_UINT &&                                                     \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_RG16_SNORM || G17P_R32_SINT || G17P_RG32_SFLOAT ||            \
    G17P_HOST_IMAGE_COPY ||                                              \
    G17P_MSAA_SAMPLES != 1)
#error "the R32 UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R32_SFLOAT &&                                                   \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_RG16_SNORM || G17P_R32_UINT || G17P_R32_SINT ||               \
    G17P_RG32_SFLOAT || G17P_HOST_IMAGE_COPY ||                         \
    G17P_MSAA_SAMPLES != 1)
#error "the R32 SFLOAT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R32_SINT &&                                                     \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_RG16_SNORM || G17P_R32_UINT || G17P_R32_SFLOAT ||             \
    G17P_RG32_SFLOAT || G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R32 SINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG16_UINT &&                                                    \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_RG16_SNORM || G17P_R32_UINT || G17P_R32_SFLOAT ||             \
    G17P_R32_SINT || G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG16 UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG16_SINT &&                                                    \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_RG16_SNORM || G17P_R32_UINT || G17P_R32_SFLOAT ||             \
    G17P_R32_SINT || G17P_RG16_UINT || G17P_HOST_IMAGE_COPY ||         \
    G17P_MSAA_SAMPLES != 1)
#error "the RG16 SINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG32_SINT &&                                                    \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_RG16_SNORM || G17P_R32_UINT || G17P_R32_SFLOAT ||             \
    G17P_R32_SINT || G17P_RG16_UINT || G17P_RG16_SINT ||               \
    G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG32 SINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG32_SFLOAT &&                                                  \
   (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG32 SFLOAT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG8_UINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG8 UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG32_UINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG32 UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG8_SINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG8 SINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if (G17P_RGBA8_UINT || G17P_RGBA8_SINT) &&                           \
   (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RGBA8 integer fixtures are guarded single-sample GPU-copy oracles"
#endif

#if G17P_RGBA8_SNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RGBA8 SNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_BGRA8_SNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the BGRA8 SNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if (G17P_BGRA8_UNORM || G17P_BGRA8_SRGB) &&                         \
   (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the BGRA8 fixtures are guarded single-sample GPU-copy oracles"
#endif

#if G17P_RG8_SNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG8 SNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R8_UINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R8 UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R8_UNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R8 UNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R8_SNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R8 SNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R8_SINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R8 SINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RGBA16_SNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RGBA16 SNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R16_SNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R16 SNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R16_UNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R16 UNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R16_UINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R16 UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R16_SINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R16 SINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R16_SFLOAT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R16 SFLOAT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG16_SFLOAT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG16 SFLOAT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RGBA16_SFLOAT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RGBA16 SFLOAT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RGBA16_UINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RGBA16 UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RGBA16_SINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RGBA16 SINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RGBA32_UINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RGBA32 UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RGBA32_SINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RGBA32 SINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_A2B10G10R10_UINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the packed UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_A2R10G10B10_UNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the packed UNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RGBA32_SFLOAT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RGBA32 SFLOAT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_TEXTURE_ALPHA_COMPOSITE && G17P_COMPOSITE_DESCRIPTOR_CELL == 1 && \
   G17P_COMPOSITE_ARRAY_NEAREST_EXACT
#define PROBE_NAME "g17p-vulkan-texture-alpha-composite-allocated-combined-array-exact"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TEXTURE_ALPHA_COMPOSITE && G17P_COMPOSITE_DESCRIPTOR_CELL == 2 && \
   G17P_COMPOSITE_ARRAY_NEAREST_EXACT
#define PROBE_NAME "g17p-vulkan-texture-alpha-composite-allocated-separate-array-exact"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TEXTURE_ALPHA_COMPOSITE && G17P_COMPOSITE_DESCRIPTOR_CELL == 1 && \
   G17P_COMPOSITE_DESCRIPTOR_ARRAY
#define PROBE_NAME "g17p-vulkan-texture-alpha-composite-allocated-combined-array"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TEXTURE_ALPHA_COMPOSITE && G17P_COMPOSITE_DESCRIPTOR_CELL == 2 && \
   G17P_COMPOSITE_DESCRIPTOR_ARRAY
#define PROBE_NAME "g17p-vulkan-texture-alpha-composite-allocated-separate-array"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TEXTURE_ALPHA_COMPOSITE && G17P_COMPOSITE_DESCRIPTOR_CELL == 1
#define PROBE_NAME "g17p-vulkan-texture-alpha-composite-allocated-combined"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TEXTURE_ALPHA_COMPOSITE && G17P_COMPOSITE_DESCRIPTOR_CELL == 2
#define PROBE_NAME "g17p-vulkan-texture-alpha-composite-allocated-separate"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TEXTURE_ALPHA_COMPOSITE && G17P_COMPOSITE_GENERATE_MIPS
#if G17P_COMPOSITE_GENERATE_MIPS_ODD
#define PROBE_NAME "g17p-vulkan-texture-alpha-composite-bgra8p-generated-mips-odd"
#else
#define PROBE_NAME "g17p-vulkan-texture-alpha-composite-bgra8p-generated-mips"
#endif
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#define GENERATED_LEVEL0_WIDTH                                           \
   (2 * IMAGE_WIDTH + G17P_COMPOSITE_GENERATE_MIPS_ODD)
#define GENERATED_LEVEL0_HEIGHT                                          \
   (2 * IMAGE_HEIGHT + G17P_COMPOSITE_GENERATE_MIPS_ODD)
#define GENERATED_LEVEL0_BYTES                                           \
   (GENERATED_LEVEL0_WIDTH * GENERATED_LEVEL0_HEIGHT * TEXTURE_PIXEL_BYTES)
#define GENERATED_PRE_LEVEL0_OFFSET READBACK_GUARD_BYTES
#define GENERATED_POST_LEVEL0_OFFSET                                        \
   (READBACK_GUARD_BYTES + GENERATED_LEVEL0_BYTES + READBACK_GUARD_BYTES)
#define GENERATED_LEVEL1_OFFSET                                             \
   (GENERATED_POST_LEVEL0_OFFSET + GENERATED_LEVEL0_BYTES +                \
    READBACK_GUARD_BYTES)
#define GENERATED_READBACK_BYTES                                            \
   (GENERATED_LEVEL1_OFFSET + IMAGE_BYTES + READBACK_GUARD_BYTES)
#elif G17P_TEXTURE_ALPHA_COMPOSITE && G17P_COMPOSITE_SRGB_SOURCE_BGRA
#define PROBE_NAME "g17p-vulkan-texture-alpha-composite-bgra8-srgb"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TEXTURE_ALPHA_COMPOSITE && G17P_COMPOSITE_SRGB_SOURCE
#define PROBE_NAME "g17p-vulkan-texture-alpha-composite-rgba8-srgb"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TEXTURE_ALPHA_COMPOSITE
#define PROBE_NAME "g17p-vulkan-texture-alpha-composite"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TWO_SAMPLERS && G17P_TWO_SAMPLERS_SPLIT_SETS
#define PROBE_NAME "g17p-vulkan-texture-two-samplers-two-sets"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TWO_SAMPLERS
#define PROBE_NAME "g17p-vulkan-texture-two-samplers-one-set"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_IMPLICIT_LOD
#define PROBE_NAME "g17p-vulkan-implicit-lod"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_EXPLICIT_GRADIENT
#define PROBE_NAME "g17p-vulkan-texture-grad"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TEXTURE_GATHER
#define PROBE_NAME "g17p-vulkan-texture-gather"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_SATURATE_TEXTURE
#define PROBE_NAME "g17p-vulkan-saturate-texture"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_SHADOW_COMPARE
#define PROBE_NAME "g17p-vulkan-shadow-compare"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TEXTURE_MIN_LOD
#define PROBE_NAME "g17p-vulkan-texture-min-lod"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TEXEL_FETCH_LOD
#define PROBE_NAME "g17p-vulkan-texel-fetch-lod"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_CUBE_SAMPLE
#define PROBE_NAME "g17p-vulkan-cube-sample"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_VERTEX_FORMAT == 1 && G17P_DYNAMIC_VERTEX_INPUT
#define PROBE_NAME "g17p-vulkan-dynamic-vertex-input-rgba8-unorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_VERTEX_FORMAT == 2 && G17P_DYNAMIC_VERTEX_INPUT
#define PROBE_NAME "g17p-vulkan-dynamic-vertex-input-rg16-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_VERTEX_FORMAT == 1
#define PROBE_NAME "g17p-vulkan-vertex-rgba8-unorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_VERTEX_FORMAT == 2
#define PROBE_NAME "g17p-vulkan-vertex-rg16-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_VERTEX_FORMAT == 3 && G17P_DYNAMIC_VERTEX_INPUT
#define PROBE_NAME "g17p-vulkan-dynamic-vertex-input-rg16-sfloat"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_VERTEX_FORMAT == 3
#define PROBE_NAME "g17p-vulkan-vertex-rg16-sfloat"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_INSTANCE_RATE
#define PROBE_NAME "g17p-vulkan-instance-rate"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_INDEX_RESTART
#define PROBE_NAME "g17p-vulkan-index-restart"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_INDIRECT_DRAW == 1
#define PROBE_NAME "g17p-vulkan-draw-indirect"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_INDIRECT_DRAW == 2
#define PROBE_NAME "g17p-vulkan-draw-indexed-indirect"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_INDIRECT_DRAW == 3
#define PROBE_NAME "g17p-vulkan-draw-indirect-count"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_INDIRECT_DRAW == 4
#define PROBE_NAME "g17p-vulkan-draw-indexed-indirect-count"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_ROBUST_INDEXED
#define PROBE_NAME "g17p-vulkan-draw-indexed-indirect-robust"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_A2R10G10B10_UNORM
#define PROBE_NAME "g17p-vulkan-a2r10g10b10-unorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA32_SFLOAT
#define PROBE_NAME "g17p-vulkan-rgba32-sfloat"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_A2B10G10R10_UINT
#define PROBE_NAME "g17p-vulkan-a2b10g10r10-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA16_SFLOAT
#define PROBE_NAME "g17p-vulkan-rgba16-sfloat"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG16_SFLOAT
#define PROBE_NAME "g17p-vulkan-rg16-sfloat"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R16_SFLOAT
#define PROBE_NAME "g17p-vulkan-r16-sfloat"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R16_SINT
#define PROBE_NAME "g17p-vulkan-r16-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R16_UINT
#define PROBE_NAME "g17p-vulkan-r16-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA32_SINT
#define PROBE_NAME "g17p-vulkan-rgba32-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA32_UINT
#define PROBE_NAME "g17p-vulkan-rgba32-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA16_SINT
#define PROBE_NAME "g17p-vulkan-rgba16-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA16_UINT
#define PROBE_NAME "g17p-vulkan-rgba16-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R16_UNORM
#define PROBE_NAME "g17p-vulkan-r16-unorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R16_SNORM
#define PROBE_NAME "g17p-vulkan-r16-snorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA16_SNORM
#define PROBE_NAME "g17p-vulkan-rgba16-snorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R8_SINT
#define PROBE_NAME "g17p-vulkan-r8-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R8_UNORM
#define PROBE_NAME "g17p-vulkan-r8-unorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R8_SNORM
#define PROBE_NAME "g17p-vulkan-r8-snorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R8_UINT
#define PROBE_NAME "g17p-vulkan-r8-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG8_SNORM
#define PROBE_NAME "g17p-vulkan-rg8-snorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_BGRA8_UNORM
#define PROBE_NAME "g17p-vulkan-bgra8-unorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_BGRA8_SRGB
#define PROBE_NAME "g17p-vulkan-bgra8-srgb"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_BGRA8_SNORM
#define PROBE_NAME "g17p-vulkan-bgra8-snorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA8_SNORM
#define PROBE_NAME "g17p-vulkan-rgba8-snorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA8_SINT
#define PROBE_NAME "g17p-vulkan-rgba8-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA8_UINT
#define PROBE_NAME "g17p-vulkan-rgba8-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG8_SINT
#define PROBE_NAME "g17p-vulkan-rg8-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG8_UINT
#define PROBE_NAME "g17p-vulkan-rg8-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA16_UNORM
#define PROBE_NAME "g17p-vulkan-rgba16-unorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG16_SNORM
#define PROBE_NAME "g17p-vulkan-rg16-snorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R32_SINT
#define PROBE_NAME "g17p-vulkan-r32-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG32_SINT
#define PROBE_NAME "g17p-vulkan-rg32-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG32_SFLOAT
#define PROBE_NAME "g17p-vulkan-rg32-sfloat"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R32_UINT
#define PROBE_NAME "g17p-vulkan-r32-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG32_UINT
#define PROBE_NAME "g17p-vulkan-rg32-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R32_SFLOAT
#define PROBE_NAME "g17p-vulkan-r32-sfloat"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG16_UINT
#define PROBE_NAME "g17p-vulkan-rg16-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG16_SINT
#define PROBE_NAME "g17p-vulkan-rg16-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG16_UNORM
#define PROBE_NAME "g17p-vulkan-rg16-unorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_F16_IO
#define PROBE_NAME "g17p-vulkan-f16-io"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_VERTEX_OUTPUT_68
#define PROBE_NAME "g17p-vulkan-vertex-output-68"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_MULTI_VIEWPORT == 2
#define PROBE_NAME "g17p-vulkan-multi-viewport-dynamic-viewport"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_MULTI_VIEWPORT == 3
#define PROBE_NAME "g17p-vulkan-multi-viewport-dynamic-scissor"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_MULTI_VIEWPORT == 4
#define PROBE_NAME "g17p-vulkan-multi-viewport-dynamic-both"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_MULTI_VIEWPORT
#define PROBE_NAME "g17p-vulkan-multi-viewport"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_SHADER_QUAD_CONTROL
#define PROBE_NAME "g17p-vulkan-shader-quad-control"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_LAYER_OUTPUT
#define PROBE_NAME "g17p-vulkan-layer-output"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (READBACK_PAYLOAD_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_MULTIVIEW
#define PROBE_NAME "g17p-vulkan-multiview"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (READBACK_PAYLOAD_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_SAMPLE_RATE_SHADING
#define PROBE_NAME "g17p-vulkan-sample-rate-shading4"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_MSAA_COVERAGE == 1
#define PROBE_NAME "g17p-vulkan-dynamic-sample-mask4"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_MSAA_COVERAGE == 3
#define PROBE_NAME "g17p-vulkan-dynamic-sample-mask4-no-sample-shading"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_MSAA_COVERAGE == 2
#define PROBE_NAME "g17p-vulkan-dynamic-alpha-to-coverage4"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_MSAA_SAMPLES > 1
#define PROBE_NAME "g17p-msaa-triangle"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_TEXTURE_3D
#define PROBE_NAME "g17p-vulkan-texture-3d"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_YCBCR
#define PROBE_NAME "g17p-vulkan-ycbcr-420"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_BRESENHAM_LINES
#define PROBE_NAME "g17p-vulkan-bresenham-lines"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_LOGIC_OP_XOR
#define PROBE_NAME "g17p-vulkan-logic-op-xor"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_BLEND_CONSTANT
#define PROBE_NAME "g17p-vulkan-blend-constant"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA8_MRT
#if G17P_STK_R11G11B10_MRT
#define PROBE_NAME "g17p-vulkan-stk-r11g11b10-mrt"
#else
#define PROBE_NAME "g17p-vulkan-rgba8-mrt"
#endif
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_IMAGE1_OFFSET (2 * READBACK_GUARD_BYTES + IMAGE_BYTES)
#define READBACK_BYTES (2 * IMAGE_BYTES + 3 * READBACK_GUARD_BYTES)
#elif G17P_DYNAMIC_BLEND
#define PROBE_NAME "g17p-vulkan-dynamic-blend"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_ADVANCED_BLEND
#define PROBE_NAME "g17p-vulkan-advanced-blend"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RASTER_DISCARD
#define PROBE_NAME "g17p-vulkan-rasterizer-discard"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_SECONDARY_COMMAND_BUFFER_MULTI
#define PROBE_NAME "g17p-vulkan-secondary-command-buffer-multi"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_SECONDARY_COMMAND_BUFFER
#define PROBE_NAME "g17p-vulkan-secondary-command-buffer"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_HOST_IMAGE_COPY
#define PROBE_NAME "g17p-hostcopy-triangle"
#define READBACK_IMAGE_USAGE VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT
#define READBACK_GUARD_BYTES 0
#define READBACK_BYTES IMAGE_BYTES
#else
#define PROBE_NAME "g17p-offscreen-triangle"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 0
#define READBACK_BYTES IMAGE_BYTES
#endif

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkEnumeratePhysicalDevices);
#if G17P_F16_IO || G17P_INDIRECT_DRAW || G17P_BRESENHAM_LINES ||          \
   G17P_MULTIVIEW || G17P_LAYER_OUTPUT || G17P_DYNAMIC_BLEND ||           \
   G17P_ADVANCED_BLEND || G17P_MULTI_VIEWPORT ||                           \
   G17P_TEXTURE_ALPHA_COMPOSITE || G17P_SAMPLE_RATE_SHADING ||            \
   G17P_MSAA_COVERAGE || G17P_SHADER_QUAD_CONTROL || G17P_YCBCR ||        \
   G17P_DYNAMIC_VERTEX_INPUT || G17P_TEXTURE_MIN_LOD
VK_FN(vkGetPhysicalDeviceFeatures2);
#endif
#if G17P_INDIRECT_DRAW || G17P_SHADER_QUAD_CONTROL || G17P_DYNAMIC_VERTEX_INPUT
VK_FN(vkEnumerateDeviceExtensionProperties);
#endif
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceMemoryProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkGetPhysicalDeviceFormatProperties);
VK_FN(vkCreateDevice);
VK_FN(vkGetDeviceQueue);
VK_FN(vkCreateBuffer);
VK_FN(vkGetBufferMemoryRequirements);
VK_FN(vkCreateImage);
VK_FN(vkGetImageMemoryRequirements);
VK_FN(vkAllocateMemory);
VK_FN(vkBindBufferMemory);
VK_FN(vkBindImageMemory);
VK_FN(vkMapMemory);
VK_FN(vkCreateImageView);
#if G17P_TEXTURE_ORACLE
VK_FN(vkCreateSampler);
#if G17P_YCBCR
VK_FN(vkCreateSamplerYcbcrConversion);
#endif
VK_FN(vkCreateDescriptorSetLayout);
VK_FN(vkCreateDescriptorPool);
VK_FN(vkAllocateDescriptorSets);
VK_FN(vkUpdateDescriptorSets);
#endif
VK_FN(vkCreateRenderPass);
VK_FN(vkCreateShaderModule);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkCreateGraphicsPipelines);
VK_FN(vkCreateFramebuffer);
VK_FN(vkCreateCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkCmdBeginRenderPass);
#if G17P_SECONDARY_COMMAND_BUFFER
VK_FN(vkCmdExecuteCommands);
#endif
#if G17P_SECONDARY_COMMAND_BUFFER_MULTI
VK_FN(vkCmdPushConstants);
#endif
#if G17P_TEXTURE_ORACLE || G17P_RGBA8_MRT
VK_FN(vkCmdPipelineBarrier);
#endif
#if G17P_TEXTURE_ORACLE
#if G17P_SHADOW_COMPARE
VK_FN(vkCmdClearDepthStencilImage);
#else
VK_FN(vkCmdCopyBufferToImage);
#endif
#if G17P_COMPOSITE_GENERATE_MIPS
VK_FN(vkCmdBlitImage);
#endif
VK_FN(vkCmdBindDescriptorSets);
#endif
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdBindVertexBuffers);
VK_FN(vkCmdDraw);
#if G17P_DYNAMIC_BLEND || G17P_TEXTURE_ALPHA_COMPOSITE ||                 \
   G17P_MULTI_VIEWPORT_DYNAMIC_SCISSOR || G17P_MSAA_COVERAGE == 2 ||     \
   G17P_MSAA_COVERAGE == 3
VK_FN(vkCmdSetScissor);
#endif
#if G17P_MULTI_VIEWPORT_DYNAMIC_VIEWPORT
VK_FN(vkCmdSetViewport);
#endif
#if G17P_DYNAMIC_BLEND
VK_FN(vkCmdSetColorBlendEnableEXT);
VK_FN(vkCmdSetColorBlendEquationEXT);
VK_FN(vkCmdSetColorWriteMaskEXT);
#endif
#if G17P_DYNAMIC_VERTEX_INPUT
VK_FN(vkCmdSetVertexInputEXT);
#endif
#if G17P_MSAA_COVERAGE == 1 || G17P_MSAA_COVERAGE == 3
VK_FN(vkCmdSetSampleMaskEXT);
#elif G17P_MSAA_COVERAGE == 2
VK_FN(vkCmdSetAlphaToCoverageEnableEXT);
#endif
#if G17P_INDIRECT_DRAW
VK_FN(vkCmdDrawIndirect);
VK_FN(vkCmdDrawIndexedIndirect);
#if G17P_INDIRECT_COUNT
VK_FN(vkCmdDrawIndirectCount);
VK_FN(vkCmdDrawIndexedIndirectCount);
#endif
#endif
#if G17P_INDIRECT_DRAW || G17P_INDEX_RESTART || \
   G17P_TEXTURE_ALPHA_COMPOSITE
VK_FN(vkCmdBindIndexBuffer);
#endif
#if G17P_INDEX_RESTART || G17P_TEXTURE_ALPHA_COMPOSITE
VK_FN(vkCmdDrawIndexed);
#endif
VK_FN(vkCmdEndRenderPass);
#if G17P_HOST_IMAGE_COPY
VK_FN(vkCopyImageToMemoryEXT);
#else
VK_FN(vkCmdCopyImageToBuffer);
#endif
#if G17P_TEXTURE_3D
VK_FN(vkCopyMemoryToImageEXT);
VK_FN(vkTransitionImageLayoutEXT);
#endif
VK_FN(vkEndCommandBuffer);
VK_FN(vkCreateFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);

#if G17P_CUBE_SAMPLE
static int cube_face_diagnostic = -1;
#endif

static void
die(const char *what)
{
   fprintf(stderr, PROBE_NAME ": %s\n", what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, PROBE_NAME ": %s returned %d\n", what, result);
      exit(EXIT_FAILURE);
   }
}

#define LOAD_GLOBAL(name)                                                      \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(VK_NULL_HANDLE, #name);         \
      if (!name)                                                               \
         die("loader is missing " #name);                                     \
   } while (0)

#define LOAD_INSTANCE(instance, name)                                          \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);               \
      if (!name)                                                               \
         die("instance is missing " #name);                                   \
   } while (0)

#define LOAD_DEVICE(device, name)                                              \
   do {                                                                        \
      name = (PFN_##name)vkGetDeviceProcAddr(device, #name);                   \
      if (!name)                                                               \
         die("device is missing " #name);                                     \
   } while (0)

static uint32_t
find_memory_type(const VkPhysicalDeviceMemoryProperties *properties,
                 uint32_t allowed, VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((allowed & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }

   die("required Vulkan memory type is unavailable");
   return UINT32_MAX;
}

static uint64_t
fnv1a64(const uint8_t *bytes, size_t size)
{
   uint64_t hash = UINT64_C(14695981039346656037);
   for (size_t i = 0; i < size; ++i) {
      hash ^= bytes[i];
      hash *= UINT64_C(1099511628211);
   }
   return hash;
}

#if G17P_INDIRECT_DRAW || G17P_SHADER_QUAD_CONTROL || G17P_DYNAMIC_VERTEX_INPUT
static bool
has_device_extension(VkPhysicalDevice physical, const char *name)
{
   uint32_t count = 0;
   check(vkEnumerateDeviceExtensionProperties(physical, NULL, &count, NULL),
         "vkEnumerateDeviceExtensionProperties(count)");
   VkExtensionProperties *extensions = calloc(count, sizeof(*extensions));
   if (!extensions)
      die("out of host memory");
   check(vkEnumerateDeviceExtensionProperties(physical, NULL, &count,
                                               extensions),
         "vkEnumerateDeviceExtensionProperties(list)");

   bool found = false;
   for (uint32_t i = 0; i < count; ++i)
      found |= strcmp(extensions[i].extensionName, name) == 0;
   free(extensions);
   return found;
}
#endif

#if G17P_INDIRECT_DRAW || G17P_INDEX_RESTART || G17P_INSTANCE_RATE || \
   G17P_VERTEX_FORMAT || G17P_TEXTURE_ORACLE
struct mapped_buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   uint8_t *map;
   size_t size;
};

static struct mapped_buffer
create_mapped_buffer(VkDevice device,
                     const VkPhysicalDeviceMemoryProperties *memory_properties,
                     size_t size, VkBufferUsageFlags usage, const char *name)
{
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   struct mapped_buffer out = {.size = size};
   check(vkCreateBuffer(device, &buffer_info, NULL, &out.buffer), name);

   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, out.buffer, &requirements);
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = find_memory_type(
         memory_properties, requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   check(vkAllocateMemory(device, &memory_info, NULL, &out.memory), name);
   check(vkBindBufferMemory(device, out.buffer, out.memory, 0), name);
   check(vkMapMemory(device, out.memory, 0, size, 0, (void **)&out.map), name);
   return out;
}

static bool
verify_upload(const struct mapped_buffer *buffer, const uint8_t *expected,
              const char *name)
{
   for (size_t i = 0; i < buffer->size; ++i) {
      if (buffer->map[i] != expected[i]) {
         fprintf(stderr,
                 PROBE_NAME ": %s byte %zu changed: got 0x%02x, expected 0x%02x\n",
                 name, i, buffer->map[i], expected[i]);
         return false;
      }
   }
   return true;
}
#endif

#if G17P_SECONDARY_COMMAND_BUFFER_MULTI
/* Quadrant 0 is replayed first, exactly as vkQuake replays its six world
 * contexts before entities, sky, the view model and the GUI.  Every channel
 * is an exact UNORM8 code point of an exact 0.0/1.0 fragment colour. */
static const float secondary_quadrant_colors[SECONDARY_QUADRANT_COUNT][4] = {
   {1.0f, 0.0f, 0.0f, 1.0f},
   {0.0f, 1.0f, 0.0f, 1.0f},
   {0.0f, 0.0f, 1.0f, 1.0f},
   {1.0f, 1.0f, 0.0f, 1.0f},
};
static const uint8_t
   secondary_quadrant_expected[SECONDARY_QUADRANT_COUNT][4] = {
   {255, 0, 0, 255},
   {0, 255, 0, 255},
   {0, 0, 255, 255},
   {255, 255, 0, 255},
};
/* The render-pass clear.  A quadrant that still carries it was never
 * rasterized, which is the vkQuake world-surface symptom rather than a
 * wrong-colour or wrong-order one. */
static const uint8_t secondary_quadrant_clear[4] = {0, 0, 0, 255};
static const char *const
   secondary_quadrant_names[SECONDARY_QUADRANT_COUNT] = {
   "0 (top-left, replayed first)",
   "1 (top-right)",
   "2 (bottom-left)",
   "3 (bottom-right, replayed last)",
};
#endif

static bool
verify_readback(const uint8_t *map)
{
#if G17P_SECONDARY_COMMAND_BUFFER_MULTI
   for (size_t i = 0; i < READBACK_GUARD_BYTES; ++i) {
      if (map[i] != 0xa5 ||
          map[READBACK_GUARD_BYTES + READBACK_PAYLOAD_BYTES + i] != 0xa5) {
         fprintf(stderr, PROBE_NAME ": readback guard %zu was modified\n", i);
         return false;
      }
   }

   const uint8_t *pixels = map + READBACK_GUARD_BYTES;

   /* Report every quadrant centre on both outcomes: a pass and a "the first
    * group is missing" failure must be equally legible from the log. */
   for (unsigned quadrant = 0; quadrant < SECONDARY_QUADRANT_COUNT;
        ++quadrant) {
      const unsigned x =
         (quadrant & 1) * (IMAGE_WIDTH / 2) + IMAGE_WIDTH / 4;
      const unsigned y =
         (quadrant >> 1) * (IMAGE_HEIGHT / 2) + IMAGE_HEIGHT / 4;
      const uint8_t *centre =
         &pixels[(y * IMAGE_WIDTH + x) * IMAGE_PIXEL_BYTES];
      printf(PROBE_NAME ": quadrant %s centre (%u,%u) RGBA = %u %u %u %u, "
                        "expected %u %u %u %u\n",
             secondary_quadrant_names[quadrant], x, y, centre[0], centre[1],
             centre[2], centre[3], secondary_quadrant_expected[quadrant][0],
             secondary_quadrant_expected[quadrant][1],
             secondary_quadrant_expected[quadrant][2],
             secondary_quadrant_expected[quadrant][3]);
   }

   size_t wrong_pixels[SECONDARY_QUADRANT_COUNT] = {0};
   size_t first_wrong_pixel[SECONDARY_QUADRANT_COUNT] = {0};
   uint8_t first_wrong[SECONDARY_QUADRANT_COUNT][IMAGE_PIXEL_BYTES];
   memset(first_wrong, 0, sizeof(first_wrong));

   for (size_t pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; ++pixel) {
      const unsigned x = pixel % IMAGE_WIDTH;
      const unsigned y = pixel / IMAGE_WIDTH;
      const unsigned quadrant =
         (x >= IMAGE_WIDTH / 2 ? 1u : 0u) + (y >= IMAGE_HEIGHT / 2 ? 2u : 0u);
      const uint8_t *actual = &pixels[pixel * IMAGE_PIXEL_BYTES];
      const uint8_t *wanted = secondary_quadrant_expected[quadrant];
      unsigned worst_delta = 0;
      for (unsigned byte = 0; byte < IMAGE_PIXEL_BYTES; ++byte) {
         const unsigned delta = actual[byte] > wanted[byte]
                                   ? actual[byte] - wanted[byte]
                                   : wanted[byte] - actual[byte];
         if (delta > worst_delta)
            worst_delta = delta;
      }
      if (worst_delta <= SECONDARY_QUADRANT_TOLERANCE)
         continue;
      if (!wrong_pixels[quadrant]) {
         first_wrong_pixel[quadrant] = pixel;
         memcpy(first_wrong[quadrant], actual, IMAGE_PIXEL_BYTES);
      }
      wrong_pixels[quadrant]++;
   }

   unsigned wrong_quadrants = 0;
   bool cleared[SECONDARY_QUADRANT_COUNT] = {false};
   for (unsigned quadrant = 0; quadrant < SECONDARY_QUADRANT_COUNT;
        ++quadrant) {
      if (!wrong_pixels[quadrant])
         continue;
      wrong_quadrants++;
      cleared[quadrant] = memcmp(first_wrong[quadrant],
                                 secondary_quadrant_clear,
                                 IMAGE_PIXEL_BYTES) == 0;
      fprintf(stderr,
              PROBE_NAME ": quadrant %s is wrong in %zu of %u pixels: pixel "
                         "%zu is RGBA %u %u %u %u, expected %u %u %u %u "
                         "+/- %u%s\n",
              secondary_quadrant_names[quadrant], wrong_pixels[quadrant],
              (unsigned)((IMAGE_WIDTH / 2) * (IMAGE_HEIGHT / 2)),
              first_wrong_pixel[quadrant], first_wrong[quadrant][0],
              first_wrong[quadrant][1], first_wrong[quadrant][2],
              first_wrong[quadrant][3],
              secondary_quadrant_expected[quadrant][0],
              secondary_quadrant_expected[quadrant][1],
              secondary_quadrant_expected[quadrant][2],
              secondary_quadrant_expected[quadrant][3],
              SECONDARY_QUADRANT_TOLERANCE,
              cleared[quadrant]
                 ? "; that is the render-pass clear, so this secondary's draw "
                   "never rasterized"
                 : "");
   }

   if (!wrong_quadrants)
      return true;

   if (wrong_quadrants == 1 && cleared[0]) {
      fprintf(stderr,
              PROBE_NAME ": FIRST GROUP MISSING — quadrant 0 kept the clear "
                         "while quadrants 1..3 painted correctly, so only the "
                         "first of four consecutive one-per-call "
                         "vkCmdExecuteCommands replays was dropped. That is "
                         "exactly the vkQuake world-surface symptom: the "
                         "world context is the first group in the pass.\n");
   }

   return false;
#elif G17P_RGBA8_MRT
   for (size_t i = 0; i < READBACK_GUARD_BYTES; ++i) {
      if (map[i] != 0xa5 ||
          map[READBACK_GUARD_BYTES + IMAGE_BYTES + i] != 0xa5 ||
          map[READBACK_IMAGE1_OFFSET + IMAGE_BYTES + i] != 0xa5) {
         fprintf(stderr, PROBE_NAME ": readback guard %zu was modified\n", i);
         return false;
      }
   }

#if G17P_STK_R11G11B10_MRT
   /* Exact B10G11R11 UFLOAT encodings. The left half receives two additive
    * writes on RT0 and one write on RT1; the right half keeps its clear. */
   static const uint8_t clear[2][4] = {
      {0xc0, 0x02, 0x18, 0x64},
      {0x50, 0x03, 0x1b, 0x6e},
   };
   static const uint8_t additive[2][4] = {
      {0x80, 0x03, 0x1a, 0x78},
      {0x00, 0x03, 0x1a, 0x58},
   };
#else
   static const uint8_t clear[2][4] = {
      {16, 32, 48, 64},
      {80, 96, 112, 128},
   };
   static const uint8_t additive[2][4] = {
      {128, 64, 255, 128},
      {32, 64, 16, 128},
   };
#endif
   const uint8_t *images[2] = {
      map + READBACK_GUARD_BYTES,
      map + READBACK_IMAGE1_OFFSET,
   };

   for (unsigned rt = 0; rt < 2; ++rt) {
      for (size_t pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; ++pixel) {
         const uint8_t *wanted =
            (pixel % IMAGE_WIDTH) < (IMAGE_WIDTH / 2) ? additive[rt]
                                                       : clear[rt];
         const uint8_t *actual = images[rt] + pixel * 4;
         if (memcmp(actual, wanted, 4) != 0) {
            fprintf(stderr, PROBE_NAME
                            ": attachment %u pixel %zu differs: got "
                            "%02x%02x%02x%02x, expected %02x%02x%02x%02x\n",
                    rt, pixel, actual[0], actual[1], actual[2], actual[3],
                    wanted[0], wanted[1], wanted[2], wanted[3]);
            return false;
         }
      }
   }

   return true;
#else
   for (size_t i = 0; i < READBACK_GUARD_BYTES; ++i) {
      if (map[i] != 0xa5 ||
          map[READBACK_GUARD_BYTES + READBACK_PAYLOAD_BYTES + i] != 0xa5) {
         fprintf(stderr, PROBE_NAME ": readback guard %zu was modified\n", i);
         return false;
      }
   }

   const uint8_t *pixels = map + READBACK_GUARD_BYTES;
#if G17P_TWO_SAMPLERS
   /* Report the centre pixel on every verification, not only on failure: a
    * passing product and a black one must both be legible from the log. */
   {
      const uint8_t *centre =
         pixels + ((IMAGE_HEIGHT / 2) * IMAGE_WIDTH + IMAGE_WIDTH / 2) *
                     IMAGE_PIXEL_BYTES;
      printf(PROBE_NAME ": centre pixel (%u,%u) RGBA = %u %u %u %u\n",
             IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, centre[0], centre[1],
             centre[2], centre[3]);
   }
#endif
#if G17P_TEXTURE_GATHER
   /* For P=(0.5,0.5) on mip 0, gather returns the red component from
    * (31,32), (32,32), (32,31), (31,31), in that order. */
   static const uint8_t expected[4] = {170, 255, 85, 0};
#elif G17P_SATURATE_TEXTURE
   /* White multiplied by (-1, 2, 0.5, 0.25), then clamped to [0, 1]. */
   static const uint8_t expected[4] = {0, 255, 128, 64};
#elif G17P_SHADOW_COMPARE
   /* With depth 0.5 and LESS, reference 0.125 passes on the left while
    * reference 0.75 fails on the right. */
   static const uint8_t expected[2][4] = {
      {255, 255, 255, 255},
      {0, 0, 0, 0},
   };
#elif G17P_TEXTURE_MIN_LOD
   /* 64 texels over 32 pixels gives implicit/gradient LOD1; txb subtracts
    * one and txl requests zero. Clamping to two must select blue instead. */
#if G17P_TEXTURE_MIN_LOD >= 5
   const unsigned level = G17P_TEXTURE_MIN_LOD - 5;
   const bool oob = level < (unsigned)view_min_lod;
   const uint8_t expected[4] = { !oob && level == 0 ? 255 : 0,
                                 !oob && level == 1 ? 255 : 0,
                                 0, oob ? 0 : 255 };
#else
   const unsigned level = view_min_lod == 2 ? 2 :
                          (G17P_TEXTURE_MIN_LOD == 1 ||
                           G17P_TEXTURE_MIN_LOD == 4) ? 1 : 0;
   const uint8_t expected[4] = {level == 0 ? 255 : 0,
                                level == 1 ? 255 : 0,
                                level == 2 ? 255 : 0, 255};
#endif
#elif G17P_TEXEL_FETCH_LOD
   /* Mip 2 is blue. Mip 0 red and mip 1 green are exact negative controls
    * for a dropped or mis-typed integer level. */
   static const uint8_t expected[4] = {0, 0, 255, 255};
#elif G17P_CUBE_SAMPLE
   static const uint8_t expected[6][4] = {
      {255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255},
      {255, 255, 0, 255}, {255, 0, 255, 255}, {0, 255, 255, 255},
   };
   static const uint8_t clear[4] = {0, 0, 0, 255};
#elif G17P_TEXTURE_3D
   /* LOD1 is 32x32x2; z=0.75 addresses its blue slice 1.  Its red slice 0
    * catches dropped or 2D-reinterpreted Z. */
   static const uint8_t expected[4] = {0, 0, 255, 255};
#elif G17P_YCBCR
   /* RGB_IDENTITY exposes the conversion's raw Cr-Y-Cb order. Distinct
    * constants in both planes catch a missing plane or swapped RG chroma. */
   static const uint8_t expected[4] = {204, 51, 102, 255};
#elif G17P_TEXTURE_ALPHA_COMPOSITE
   /* Descriptor 1 mip 1 is a transparent/green checker. A quarter-texel
    * offset makes linear filtering produce 0, 2/8, 3/8, 5/8 or 6/8
    * coverage. Seventeen scissored draws sweep opacity 0/16..16/16. With
    * source (0,192,0) and clear (128,64,128), every expected UNORM8 channel
    * is an exact integer: product = coverage_eighths * opacity_sixteenths,
    * colour = (128-product, 64+product, 128-product, 255). */
#elif G17P_TWO_SAMPLERS
   /* Both sampled textures are uniform, so every covered pixel carries the
    * same product. A shader that only ever receives its first texture leaves
    * RGB at zero here, which is exactly the vkQuake world-surface failure. */
   static const uint8_t expected[4] = G17P_TWO_SAMPLERS_EXPECTED;
#elif G17P_LOD_ORACLE
   /* The vertex shader maps the visible clip range to UV 0..1. A 64-wide
    * texture over 32 pixels therefore has rho=2 and selects exact mip 1.
    * The red mip 0 and blue mip 2 catch under- and overestimated LOD. */
   static const uint8_t expected[4] = {0, 255, 0, 255};
#elif G17P_A2R10G10B10_UNORM
   static const uint8_t expected[4] = {0xab, 0xab, 0x5a, 0x95};
#elif G17P_RGBA32_SFLOAT
   static const uint8_t expected[16] = {
      0x00, 0x00, 0x20, 0xbe, 0x00, 0x00, 0x5a, 0x41,
      0x00, 0x00, 0xe8, 0xc0, 0x00, 0x00, 0x00, 0x3d,
   };
#elif G17P_A2B10G10R10_UINT
   static const uint8_t expected[4] = {0x55, 0xa9, 0x3a, 0xbc};
#elif G17P_RGBA16_SFLOAT
   static const uint8_t expected[8] = {
      0x80, 0xc6, 0xa0, 0x4a, 0x55, 0x35, 0x00, 0xb0,
   };
#elif G17P_RG16_SFLOAT
   static const uint8_t expected[4] = {0x80, 0xc6, 0xa0, 0x4a};
#elif G17P_R16_SFLOAT
   static const uint8_t expected[2] = {0x80, 0xc6};
#elif G17P_R16_SINT
   static const uint8_t expected[2] = {0xbb, 0xdc};
#elif G17P_R16_UINT
   static const uint8_t expected[2] = {0xef, 0xbe};
#elif G17P_RGBA32_SINT
   static const uint8_t expected[16] = {
      0x99, 0xba, 0xdc, 0xfe, 0x67, 0x45, 0x23, 0x01,
      0xef, 0xbe, 0xad, 0xde, 0x10, 0x32, 0x54, 0x76,
   };
#elif G17P_RGBA32_UINT
   static const uint8_t expected[16] = {
      0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01,
      0xef, 0xbe, 0xad, 0xde, 0x10, 0x32, 0x54, 0x76,
   };
#elif G17P_RGBA16_SINT
   static const uint8_t expected[8] = {
      0xd7, 0xf6, 0x67, 0x45, 0xc7, 0xcf, 0x6d, 0x7d,
   };
#elif G17P_RGBA16_UINT
   static const uint8_t expected[8] = {
      0x57, 0x13, 0xe1, 0xac, 0x68, 0x24, 0xb9, 0xfd,
   };
#elif G17P_R16_UNORM
   static const uint8_t expected[2] = {0xef, 0xbe};
#elif G17P_R16_SNORM
   static const uint8_t expected[2] = {0x00, 0xc0};
#elif G17P_RGBA16_SNORM
   static const uint8_t expected[8] = {
      0x00, 0xc0, 0x01, 0x10, 0x00, 0x60, 0xff, 0xdf,
   };
#elif G17P_R8_SINT
   static const uint8_t expected[1] = {0xa3};
#elif G17P_R8_UNORM
   static const uint8_t expected[1] = {0x95};
#elif G17P_R8_SNORM
   static const uint8_t expected[1] = {0xc0};
#elif G17P_R8_UINT
   static const uint8_t expected[1] = {0xa7};
#elif G17P_RG8_SNORM
   static const uint8_t expected[2] = {0xc0, 0x11};
#elif G17P_BGRA8_UNORM
   /* The fragment writes RGBA=(1,.28,.05,1); BGRA memory order is B,G,R,A. */
   static const uint8_t expected[4] = {13, 71, 255, 255};
#elif G17P_BGRA8_SRGB
   /* IEC 61966-2-1 transfer of the linear fragment colour, rounded to UNORM:
    * R=255, G=144, B=63, then stored in BGRA byte order. */
   static const uint8_t expected[4] = {63, 144, 255, 255};
#elif G17P_LOGIC_OP_XOR
   /* The clear is exact RGBA8 55 aa 0f ff. XOR with the fragment's
    * ff 47 0d ff distinguishes the destination load from a source store. */
   static const uint8_t expected[4] = {0xaa, 0xed, 0x02, 0x00};
#elif G17P_BLEND_CONSTANT
   /* CONSTANT_COLOR/CONSTANT_ALPHA times the source, with no destination
    * contribution. These are independently rounded RGBA8 results for
    * constants (0.25, 0.5, 0.75, 0.125). */
   static const uint8_t expected[4] = {64, 36, 10, 32};
#elif G17P_DYNAMIC_BLEND
   static const uint8_t expected[2][4] = {
      /* Left: dynamic reverse-subtract blend with only R/G enabled. */
      /* The destination is first quantized to 128/255, so
       * round((128/255 - 0.28) * 255) = 57. */
      {0, 57, 191, 255},
      /* Right: dynamic blend disable with only B/A enabled. */
      {64, 128, 13, 255},
   };
#elif G17P_ADVANCED_BLEND
   /* With opaque source and destination, MULTIPLY reduces to src.rgb *
    * dst.rgb and source-over alpha remains one. */
   static const uint8_t expected[4] = {64, 36, 10, 255};
#elif G17P_BGRA8_SNORM
   static const uint8_t expected[4] = {0x60, 0x11, 0xc0, 0xdf};
#elif G17P_RGBA8_SNORM
   static const uint8_t expected[4] = {0xc0, 0x11, 0x60, 0xdf};
#elif G17P_RGBA8_SINT
   static const uint8_t expected[4] = {0x93, 0x64, 0xcc, 0x21};
#elif G17P_RGBA8_UINT
   static const uint8_t expected[4] = {0x13, 0x57, 0xac, 0xe1};
#elif G17P_RG8_SINT
   static const uint8_t expected[2] = {0xdb, 0x64};
#elif G17P_RG8_UINT
   static const uint8_t expected[2] = {0x57, 0xac};
#elif G17P_R32_SINT
   /* Exact signed value -19088743 is 0xfedcba99 in two's complement. */
   static const uint8_t expected[4] = {0x99, 0xba, 0xdc, 0xfe};
#elif G17P_RG32_SINT
   /* Exact signed pair -19088743 and 19088743. */
   static const uint8_t expected[8] = {
      0x99, 0xba, 0xdc, 0xfe, 0x67, 0x45, 0x23, 0x01,
   };
#elif G17P_RG16_UINT
   static const uint8_t expected[4] = {0x57, 0x13, 0xe1, 0xac};
#elif G17P_RG16_SINT
   static const uint8_t expected[4] = {0xd7, 0xf6, 0x67, 0x45};
#elif G17P_R32_UINT
   static const uint8_t expected[4] = {0xef, 0xcd, 0xab, 0x89};
#elif G17P_RG32_UINT
   static const uint8_t expected[8] = {
      0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01,
   };
#elif G17P_R32_SFLOAT
   /* -0.15625 is exactly 0xbe200000 as binary32. */
   static const uint8_t expected[4] = {0x00, 0x00, 0x20, 0xbe};
#elif G17P_RG32_SFLOAT
   /* Both lanes are exact binary32: -0.15625 and 13.625. */
   static const uint8_t expected[8] = {
      0x00, 0x00, 0x20, 0xbe, 0x00, 0x00, 0x5a, 0x41,
   };
#elif G17P_RG16_SNORM
   /* Fragment (-0.5, 0.28) maps to signed-normalized -16384 (0xc000) and
    * 9175 (0x23d7). The negative first lane distinguishes SNORM from UNORM. */
   static const uint8_t expected[4] = {0x00, 0xc0, 0xd7, 0x23};
#elif G17P_RG16_UNORM
   static const uint8_t expected[4] = {0xff, 0xff, 0xae, 0x47};
#elif G17P_RGBA16_UNORM
   static const uint8_t expected[8] = {
      0xff, 0xff, 0xae, 0x47, 0xcd, 0x0c, 0xff, 0xff,
   };
#elif G17P_SAMPLE_RATE_SHADING
   /* Four independent SampleID invocations write red, green, blue and white;
    * the average resolve is exactly half intensity in every RGB channel. */
   static const uint8_t expected[4] = {128, 128, 128, 255};
#elif G17P_MSAA_COVERAGE == 1
   /* Dynamic mask 0b0101 retains the red and blue SampleID invocations;
    * the other two samples retain the opaque-black clear. */
   static const uint8_t expected[4] = {64, 0, 64, 255};
#elif G17P_MSAA_COVERAGE == 3
   /* One ordinary per-pixel invocation writes the same colour to every sample
    * it covers. The left half keeps all four samples, so the average resolve
    * is exactly the fragment colour; the right half masks every sample off, so
    * the opaque-black clear survives untouched. Both halves are exactly
    * representable, so this oracle does not depend on resolve rounding. */
   static const uint8_t expected[2][4] = {
      {255, 71, 13, 255},
      {0, 0, 0, 255},
   };
#elif G17P_MSAA_COVERAGE == 2
   static const uint8_t expected[2][4] = {
      {255, 0, 0, 0},
      {0, 0, 0, 255},
   };
#elif G17P_RASTER_DISCARD
   static const uint8_t expected[4] = {0, 0, 0, 255};
#elif G17P_SHADER_QUAD_CONTROL
   /* Both horizontal lanes require the inactive partner's value on the
    * odd-only second iteration. Blue also records lane parity. */
   static const uint8_t expected[2][4] = {
      {255, 255, 0, 255},
      {255, 255, 255, 255},
   };
#else
   static const uint8_t expected[4] = {255, 71, 13, 255};
#endif
#if G17P_BRESENHAM_LINES
   static const uint8_t clear[4] = {0, 0, 0, 255};
   for (size_t pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; ++pixel) {
      const unsigned y = pixel / IMAGE_WIDTH;
      const uint8_t *wanted = (y >= 15 && y <= 17) ? expected : clear;
      const uint8_t *actual = &pixels[pixel * sizeof(expected)];
      if (memcmp(actual, wanted, sizeof(expected)) != 0) {
         fprintf(stderr, PROBE_NAME ": pixel %zu differs: got "
                         "%02x%02x%02x%02x, expected %02x%02x%02x%02x\n",
                 pixel, actual[0], actual[1], actual[2], actual[3], wanted[0],
                 wanted[1], wanted[2], wanted[3]);
         return false;
      }
   }
#else
#if G17P_COMPOSITE_SRGB_SOURCE || G17P_COMPOSITE_SRGB_TARGET
   size_t srgb_mismatch_pixels = 0;
   size_t srgb_mismatch_components = 0;
   unsigned srgb_max_delta = 0;
   bool srgb_out_of_tolerance = false;
#endif
   for (size_t pixel = 0;
        pixel < IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_LAYERS; ++pixel) {
      const uint8_t *actual = &pixels[pixel * IMAGE_PIXEL_BYTES];
#if G17P_SHADOW_COMPARE
      const uint8_t *expected_pixel =
         expected[(pixel % IMAGE_WIDTH) >= (IMAGE_WIDTH / 2)];
#elif G17P_CUBE_SAMPLE
      const unsigned x = pixel % IMAGE_WIDTH;
      const unsigned face = x < 5 ? 0 : x < 10 ? 1 : x < 15 ? 2 :
                            x < 20 ? 3 : x < 25 ? 4 : 5;
      const uint8_t *expected_pixel =
         cube_face_diagnostic < 0 || cube_face_diagnostic == (int)face
            ? expected[face]
            : clear;
#elif G17P_DYNAMIC_BLEND
      const uint8_t *expected_pixel =
         expected[(pixel % IMAGE_WIDTH) >= (IMAGE_WIDTH / 2)];
#elif G17P_MSAA_COVERAGE == 2 || G17P_MSAA_COVERAGE == 3
      const uint8_t *expected_pixel =
         expected[(pixel % IMAGE_WIDTH) >= (IMAGE_WIDTH / 2)];
#elif G17P_SHADER_QUAD_CONTROL
      const uint8_t *expected_pixel = expected[(pixel % IMAGE_WIDTH) & 1];
#elif G17P_TEXTURE_ALPHA_COMPOSITE
      const unsigned x = pixel % IMAGE_WIDTH;
      const unsigned y = pixel / IMAGE_WIDTH;
      const bool covered =
         (G17P_COMPOSITE_NARROW_SINGLE || G17P_COMPOSITE_NARROW_TWO ||
          G17P_COMPOSITE_NARROW_TWO_CHANGED) ?
            (x >= G17P_COMPOSITE_NARROW_X &&
             x < G17P_COMPOSITE_NARROW_X + G17P_COMPOSITE_NARROW_WIDTH &&
             y >= G17P_COMPOSITE_NARROW_Y &&
             y < G17P_COMPOSITE_NARROW_Y +
                    (G17P_COMPOSITE_NARROW_HEIGHT
                        ? G17P_COMPOSITE_NARROW_HEIGHT
                        : IMAGE_HEIGHT)) :
         (!G17P_COMPOSITE_FRACTIONAL_INSET ||
          (x >= 4 && x < 14 && y >= 3 && y < 12));
      uint8_t expected_pixel[4];
      unsigned coverage_eighths = 0;
      if (covered) {
         if (G17P_COMPOSITE_ARRAY_NEAREST_EXACT)
            coverage_eighths = 8;
         else if (x == IMAGE_WIDTH - 1 && y == IMAGE_HEIGHT - 1)
            coverage_eighths = ((x ^ y) & 1) ? 8 : 0;
         else if (x == IMAGE_WIDTH - 1)
            coverage_eighths = ((x ^ y) & 1) ? 6 : 2;
         else if (y == IMAGE_HEIGHT - 1)
            coverage_eighths = ((x ^ y) & 1) ? 6 : 2;
         else
            coverage_eighths = ((x ^ y) & 1) ? 5 : 3;
      }
      const unsigned opacity_sixteenths =
         (G17P_COMPOSITE_SINGLE_DRAW || G17P_COMPOSITE_REPEATED_FULL) ? 16 :
         (G17P_COMPOSITE_NARROW_SINGLE || G17P_COMPOSITE_NARROW_TWO ||
          G17P_COMPOSITE_NARROW_TWO_CHANGED) ? 1 :
         (((x + 1) * G17P_COMPOSITE_OPACITY_COUNT) - 1) / IMAGE_WIDTH;
      const unsigned product = coverage_eighths * opacity_sixteenths;
#if G17P_COMPOSITE_SRGB_SOURCE || G17P_COMPOSITE_SRGB_TARGET
      /* Vulkan decodes the sampled sRGB texels before filtering and performs
       * fixed-function blending in linear space before encoding the sRGB
       * attachment. Transparent black plus an opaque checker texel remains
       * premultiplied through both linear filtering and the opacity multiply. */
      const float contribution = product / 128.0f;
      const float destination_weight = 1.0f - contribution;
      const uint8_t source_srgb[3] = {0x40, 0x80, 0xc0};
      const float destination_linear[3] = {
         128.0f / 255.0f, 64.0f / 255.0f, 128.0f / 255.0f,
      };
      uint8_t logical[4];
      for (unsigned channel = 0; channel < 3; ++channel) {
         const float source_linear =
            util_format_srgb_8unorm_to_linear_float(source_srgb[channel]);
         const float result_linear = source_linear * contribution +
                                     destination_linear[channel] *
                                        destination_weight;
         logical[channel] = G17P_COMPOSITE_SRGB_TARGET
                               ? util_format_linear_float_to_srgb_8unorm(
                                    result_linear)
                               : (uint8_t)(result_linear * 255.0f + 0.5f);
      }
      logical[3] = 255;
      if (G17P_COMPOSITE_SRGB_TARGET) {
         expected_pixel[0] = logical[2];
         expected_pixel[1] = logical[1];
         expected_pixel[2] = logical[0];
         expected_pixel[3] = logical[3];
      } else {
         memcpy(expected_pixel, logical, sizeof(expected_pixel));
      }
#else
      expected_pixel[0] = 128 - product;
      expected_pixel[1] = 64 + product;
      expected_pixel[2] = 128 - product;
      expected_pixel[3] = 255;
#endif
#else
      const uint8_t *expected_pixel = expected;
#endif
      if (memcmp(actual, expected_pixel, IMAGE_PIXEL_BYTES) != 0) {
#if G17P_COMPOSITE_SRGB_SOURCE || G17P_COMPOSITE_SRGB_TARGET
         bool pixel_out_of_tolerance = false;
         srgb_mismatch_pixels++;
         for (size_t byte = 0; byte < IMAGE_PIXEL_BYTES; ++byte) {
            const unsigned delta = actual[byte] > expected_pixel[byte]
                                      ? actual[byte] - expected_pixel[byte]
                                      : expected_pixel[byte] - actual[byte];
            if (delta) {
               srgb_mismatch_components++;
               if (delta > srgb_max_delta)
                  srgb_max_delta = delta;
            }

            /* Vulkan requires an adjacent representable UNORM value and only
             * recommends round-to-nearest. The G17P software EOTF inverse is
             * also evaluated with implementation-defined floating-point pow
             * accuracy. Accept one RGB code point; alpha has no sRGB transfer
             * and remains exact. */
            if (delta > (byte < 3 ? 1 : 0))
               pixel_out_of_tolerance = true;
         }

         if (pixel_out_of_tolerance && !srgb_out_of_tolerance) {
            fprintf(stderr, PROBE_NAME ": first out-of-tolerance pixel %zu: got",
                    pixel);
            for (size_t byte = 0; byte < IMAGE_PIXEL_BYTES; ++byte)
               fprintf(stderr, " %02x", actual[byte]);
            fprintf(stderr, ", expected");
            for (size_t byte = 0; byte < IMAGE_PIXEL_BYTES; ++byte)
               fprintf(stderr, " %02x", expected_pixel[byte]);
            fputc('\n', stderr);
         }
         srgb_out_of_tolerance |= pixel_out_of_tolerance;
#elif G17P_TWO_SAMPLERS
         unsigned worst_delta = 0;
         for (size_t byte = 0; byte < IMAGE_PIXEL_BYTES; ++byte) {
            const unsigned delta = actual[byte] > expected_pixel[byte]
                                      ? actual[byte] - expected_pixel[byte]
                                      : expected_pixel[byte] - actual[byte];
            if (delta > worst_delta)
               worst_delta = delta;
         }
         if (worst_delta > G17P_TWO_SAMPLERS_TOLERANCE) {
            fprintf(stderr,
                    PROBE_NAME ": pixel %zu RGBA %u %u %u %u is outside the "
                               "expected product %u %u %u %u +/- %u (worst "
                               "channel delta %u)%s\n",
                    pixel, actual[0], actual[1], actual[2], actual[3],
                    expected_pixel[0], expected_pixel[1], expected_pixel[2],
                    expected_pixel[3], G17P_TWO_SAMPLERS_TOLERANCE,
                    worst_delta,
                    (actual[0] | actual[1] | actual[2]) == 0
                       ? "; RGB is black, so one of the two sampled textures "
                         "returned zero"
                       : "");
            return false;
         }
#else
         fprintf(stderr, PROBE_NAME ": pixel %zu differs: got", pixel);
         for (size_t byte = 0; byte < IMAGE_PIXEL_BYTES; ++byte)
            fprintf(stderr, " %02x", actual[byte]);
         fprintf(stderr, ", expected");
         for (size_t byte = 0; byte < IMAGE_PIXEL_BYTES; ++byte)
            fprintf(stderr, " %02x", expected_pixel[byte]);
         fputc('\n', stderr);
         return false;
#endif
      }
   }
#if G17P_COMPOSITE_SRGB_SOURCE || G17P_COMPOSITE_SRGB_TARGET
   if (srgb_mismatch_pixels) {
      fprintf(stderr,
              PROBE_NAME ": sRGB quantization summary: %zu differing pixels, "
              "%zu differing components, max delta %u LSB%s\n",
              srgb_mismatch_pixels, srgb_mismatch_components, srgb_max_delta,
              srgb_out_of_tolerance ? " (OUT OF TOLERANCE)" : "");
   }
   if (srgb_out_of_tolerance)
      return false;
#endif
#endif

   return true;
#endif
}

#if G17P_COMPOSITE_GENERATE_MIPS
static bool
verify_generated_mip1(const uint8_t *map)
{
   for (size_t i = 0; i < READBACK_GUARD_BYTES; ++i) {
      if (map[i] != 0xa5 ||
          map[GENERATED_PRE_LEVEL0_OFFSET + GENERATED_LEVEL0_BYTES + i] !=
             0xa5 ||
          map[GENERATED_POST_LEVEL0_OFFSET + GENERATED_LEVEL0_BYTES + i] !=
             0xa5 ||
          map[GENERATED_LEVEL1_OFFSET + IMAGE_BYTES + i] != 0xa5) {
         fprintf(stderr, PROBE_NAME
                         ": generated-mip readback guard %zu was modified\n",
                 i);
         return false;
      }
   }

   const uint8_t transparent[4] = {0, 0, 0, 0};
   const uint8_t green[4] = {0, 192, 0, 255};
   const size_t level0_offsets[2] = {
      GENERATED_PRE_LEVEL0_OFFSET,
      GENERATED_POST_LEVEL0_OFFSET,
   };
   const char *level0_names[2] = {"pre-blit", "post-blit"};
   for (unsigned snapshot = 0; snapshot < 2; ++snapshot) {
      const uint8_t *level0 = map + level0_offsets[snapshot];
      for (unsigned y = 0; y < GENERATED_LEVEL0_HEIGHT; ++y) {
         for (unsigned x = 0; x < GENERATED_LEVEL0_WIDTH; ++x) {
            const uint8_t *expected = ((x / 2) ^ (y / 2)) & 1
                                         ? green : transparent;
            const uint8_t *actual =
               level0 + (y * GENERATED_LEVEL0_WIDTH + x) * 4;
            if (memcmp(actual, expected, 4) != 0) {
               fprintf(stderr, PROBE_NAME
                               ": %s level0 pixel (%u,%u) got "
                               "%02x %02x %02x %02x, expected "
                               "%02x %02x %02x %02x\n",
                       level0_names[snapshot], x, y,
                       actual[0], actual[1], actual[2], actual[3],
                       expected[0], expected[1], expected[2], expected[3]);
               return false;
            }
         }
      }
   }

#if G17P_COMPOSITE_GENERATE_MIPS_ODD
   uint64_t hash = UINT64_C(14695981039346656037);
   const uint8_t *pixels = map + GENERATED_LEVEL1_OFFSET;
   for (size_t i = 0; i < IMAGE_BYTES; ++i) {
      hash ^= pixels[i];
      hash *= UINT64_C(1099511628211);
   }
   if (hash != UINT64_C(0xa5df1f2ca9de2521)) {
      fprintf(stderr, PROBE_NAME
                      ": odd generated mip1 hash %016" PRIx64
                      ", expected a5df1f2ca9de2521\n",
              hash);
      return false;
   }
   return true;
#else

   const uint8_t *pixels = map + GENERATED_LEVEL1_OFFSET;
   const uint8_t gray[4] = {128, 128, 128, 128};
   const uint8_t mixed[4] = {0, 128, 128, 0};
   size_t exact_transparent = 0, exact_green = 0;
   size_t exact_gray = 0, exact_mixed = 0, other = 0;
   bool mismatch = false;
   for (unsigned y = 0; y < IMAGE_HEIGHT; ++y) {
      for (unsigned x = 0; x < IMAGE_WIDTH; ++x) {
         const uint8_t *expected = ((x ^ y) & 1) ? green : transparent;
         const uint8_t *actual = pixels + (y * IMAGE_WIDTH + x) * 4;
         exact_transparent += memcmp(actual, transparent, 4) == 0;
         exact_green += memcmp(actual, green, 4) == 0;
         exact_gray += memcmp(actual, gray, 4) == 0;
         exact_mixed += memcmp(actual, mixed, 4) == 0;
         other += memcmp(actual, transparent, 4) != 0 &&
                  memcmp(actual, green, 4) != 0 &&
                  memcmp(actual, gray, 4) != 0 &&
                  memcmp(actual, mixed, 4) != 0;
         if (memcmp(actual, expected, 4) != 0) {
            if (!mismatch)
               fprintf(stderr, PROBE_NAME
                               ": generated mip1 pixel (%u,%u) got "
                               "%02x %02x %02x %02x, expected "
                               "%02x %02x %02x %02x\n",
                       x, y, actual[0], actual[1], actual[2], actual[3],
                       expected[0], expected[1], expected[2], expected[3]);
            mismatch = true;
         }
      }
   }
   if (mismatch)
      fprintf(stderr, PROBE_NAME
                      ": mip1 histogram: transparent=%zu green=%zu "
                      "gray80808080=%zu mixed00808000=%zu other=%zu\n",
              exact_transparent, exact_green, exact_gray, exact_mixed, other);
   return !mismatch;
#endif
}
#endif

int
main(void)
{
#if G17P_CUBE_SAMPLE
   const char *cube_face = getenv("G17P_CUBE_SAMPLE_FACE");
   if (cube_face) {
      char *end = NULL;
      long parsed = strtol(cube_face, &end, 10);
      if (!end || *end != '\0' || parsed < 0 || parsed >= 6) {
         fprintf(stderr,
                 PROBE_NAME ": G17P_CUBE_SAMPLE_FACE must be 0 through 5\n");
         return EXIT_FAILURE;
      }
      cube_face_diagnostic = parsed;
   }
#endif
   const bool record_only =
      (G17P_VERTEX_OUTPUT_68 &&
       getenv("G17P_VERTEX_OUTPUT_68_RECORD_ONLY") != NULL) ||
      (G17P_TEXTURE_ALPHA_COMPOSITE &&
       getenv("G17P_TEXTURE_ALPHA_COMPOSITE_RECORD_ONLY") != NULL) ||
      (G17P_IMPLICIT_LOD &&
       getenv("G17P_IMPLICIT_LOD_RECORD_ONLY") != NULL) ||
      (G17P_EXPLICIT_GRADIENT &&
       getenv("G17P_TEXTURE_GRAD_RECORD_ONLY") != NULL) ||
      (G17P_TEXTURE_GATHER &&
       getenv("G17P_TEXTURE_GATHER_RECORD_ONLY") != NULL) ||
      (G17P_SATURATE_TEXTURE &&
       getenv("G17P_SATURATE_TEXTURE_RECORD_ONLY") != NULL) ||
      (G17P_SHADOW_COMPARE &&
       getenv("G17P_SHADOW_COMPARE_RECORD_ONLY") != NULL) ||
      (G17P_TEXEL_FETCH_LOD &&
       getenv("G17P_TEXEL_FETCH_LOD_RECORD_ONLY") != NULL) ||
      (G17P_CUBE_SAMPLE &&
       getenv("G17P_CUBE_SAMPLE_RECORD_ONLY") != NULL) ||
      (G17P_TEXTURE_3D &&
       getenv("G17P_TEXTURE_3D_RECORD_ONLY") != NULL) ||
      (G17P_YCBCR && getenv("G17P_YCBCR_RECORD_ONLY") != NULL) ||
      (G17P_VERTEX_FORMAT &&
       getenv("G17P_VERTEX_FORMAT_RECORD_ONLY") != NULL) ||
      (G17P_INSTANCE_RATE &&
       getenv("G17P_INSTANCE_RATE_RECORD_ONLY") != NULL) ||
      (G17P_INDEX_RESTART &&
       getenv("G17P_INDEX_RESTART_RECORD_ONLY") != NULL) ||
      (G17P_MSAA_SAMPLES > 1 &&
       getenv("G17P_MSAA_RECORD_ONLY") != NULL) ||
      (G17P_INDIRECT_DRAW && getenv("G17P_INDIRECT_RECORD_ONLY") != NULL) ||
      (G17P_MULTIVIEW && getenv("G17P_MULTIVIEW_RECORD_ONLY") != NULL) ||
      (G17P_LAYER_OUTPUT &&
       getenv("G17P_LAYER_OUTPUT_RECORD_ONLY") != NULL) ||
      (G17P_MULTI_VIEWPORT &&
       getenv("G17P_MULTI_VIEWPORT_RECORD_ONLY") != NULL) ||
      (G17P_SHADER_QUAD_CONTROL &&
       getenv("G17P_SHADER_QUAD_CONTROL_RECORD_ONLY") != NULL) ||
      (G17P_A2R10G10B10_UNORM &&
       getenv("G17P_A2R10G10B10_UNORM_RECORD_ONLY") != NULL) ||
      (G17P_RGBA32_SFLOAT &&
       getenv("G17P_RGBA32_SFLOAT_RECORD_ONLY") != NULL) ||
      (G17P_A2B10G10R10_UINT &&
       getenv("G17P_A2B10G10R10_UINT_RECORD_ONLY") != NULL) ||
      (G17P_RGBA16_SFLOAT &&
       getenv("G17P_RGBA16_SFLOAT_RECORD_ONLY") != NULL) ||
      (G17P_RG16_SFLOAT && getenv("G17P_RG16_SFLOAT_RECORD_ONLY") != NULL) ||
      (G17P_R16_SFLOAT && getenv("G17P_R16_SFLOAT_RECORD_ONLY") != NULL) ||
      (G17P_R16_SINT && getenv("G17P_R16_SINT_RECORD_ONLY") != NULL) ||
      (G17P_R16_UINT && getenv("G17P_R16_UINT_RECORD_ONLY") != NULL) ||
      (G17P_R16_UNORM && getenv("G17P_R16_UNORM_RECORD_ONLY") != NULL) ||
      (G17P_R16_SNORM && getenv("G17P_R16_SNORM_RECORD_ONLY") != NULL) ||
      (G17P_RGBA16_SNORM &&
       getenv("G17P_RGBA16_SNORM_RECORD_ONLY") != NULL) ||
      (G17P_R8_UNORM && getenv("G17P_R8_UNORM_RECORD_ONLY") != NULL) ||
      (G17P_R8_SNORM && getenv("G17P_R8_SNORM_RECORD_ONLY") != NULL) ||
      (G17P_RG8_SNORM && getenv("G17P_RG8_SNORM_RECORD_ONLY") != NULL) ||
      (G17P_F16_IO && getenv("G17P_F16_IO_RECORD_ONLY") != NULL) ||
      (G17P_RG16_UNORM && getenv("G17P_RG16_UNORM_RECORD_ONLY") != NULL) ||
      (G17P_RGBA16_UNORM &&
       getenv("G17P_RGBA16_UNORM_RECORD_ONLY") != NULL) ||
      (G17P_RGBA16_UINT &&
       getenv("G17P_RGBA16_UINT_RECORD_ONLY") != NULL) ||
      (G17P_RGBA16_SINT &&
       getenv("G17P_RGBA16_SINT_RECORD_ONLY") != NULL) ||
      (G17P_RGBA32_UINT &&
       getenv("G17P_RGBA32_UINT_RECORD_ONLY") != NULL) ||
      (G17P_RGBA32_SINT &&
       getenv("G17P_RGBA32_SINT_RECORD_ONLY") != NULL) ||
      (G17P_RG16_SNORM && getenv("G17P_RG16_SNORM_RECORD_ONLY") != NULL) ||
      (G17P_R32_UINT && getenv("G17P_R32_UINT_RECORD_ONLY") != NULL) ||
      (G17P_R32_SFLOAT && getenv("G17P_R32_SFLOAT_RECORD_ONLY") != NULL) ||
      (G17P_R32_SINT && getenv("G17P_R32_SINT_RECORD_ONLY") != NULL) ||
      (G17P_RG16_UINT && getenv("G17P_RG16_UINT_RECORD_ONLY") != NULL) ||
      (G17P_RG16_SINT && getenv("G17P_RG16_SINT_RECORD_ONLY") != NULL) ||
      (G17P_RG32_SINT && getenv("G17P_RG32_SINT_RECORD_ONLY") != NULL) ||
      (G17P_RG32_SFLOAT &&
       getenv("G17P_RG32_SFLOAT_RECORD_ONLY") != NULL) ||
      (G17P_RG8_UINT && getenv("G17P_RG8_UINT_RECORD_ONLY") != NULL) ||
      (G17P_RG8_SINT && getenv("G17P_RG8_SINT_RECORD_ONLY") != NULL) ||
      (G17P_RGBA8_UINT && getenv("G17P_RGBA8_UINT_RECORD_ONLY") != NULL) ||
      (G17P_RGBA8_SINT && getenv("G17P_RGBA8_SINT_RECORD_ONLY") != NULL) ||
      (G17P_BGRA8_UNORM &&
       getenv("G17P_BGRA8_UNORM_RECORD_ONLY") != NULL) ||
      (G17P_BGRA8_SRGB &&
       getenv("G17P_BGRA8_SRGB_RECORD_ONLY") != NULL) ||
      (G17P_BGRA8_SNORM && getenv("G17P_BGRA8_SNORM_RECORD_ONLY") != NULL) ||
      (G17P_RGBA8_SNORM && getenv("G17P_RGBA8_SNORM_RECORD_ONLY") != NULL) ||
      (G17P_BRESENHAM_LINES &&
       getenv("G17P_BRESENHAM_LINES_RECORD_ONLY") != NULL) ||
      (G17P_LOGIC_OP_XOR &&
       getenv("G17P_LOGIC_OP_XOR_RECORD_ONLY") != NULL) ||
      (G17P_BLEND_CONSTANT &&
       getenv("G17P_BLEND_CONSTANT_RECORD_ONLY") != NULL) ||
      (G17P_RGBA8_MRT && getenv("G17P_RGBA8_MRT_RECORD_ONLY") != NULL) ||
      (G17P_DYNAMIC_BLEND &&
       getenv("G17P_DYNAMIC_BLEND_RECORD_ONLY") != NULL) ||
      (G17P_ADVANCED_BLEND &&
       getenv("G17P_ADVANCED_BLEND_RECORD_ONLY") != NULL) ||
      (G17P_RASTER_DISCARD &&
       getenv("G17P_RASTER_DISCARD_RECORD_ONLY") != NULL) ||
      (G17P_SECONDARY_COMMAND_BUFFER &&
       getenv("G17P_SECONDARY_COMMAND_BUFFER_RECORD_ONLY") != NULL) ||
      (G17P_R8_UINT && getenv("G17P_R8_UINT_RECORD_ONLY") != NULL) ||
      (G17P_R8_SINT && getenv("G17P_R8_SINT_RECORD_ONLY") != NULL) ||
      (G17P_RG32_UINT && getenv("G17P_RG32_UINT_RECORD_ONLY") != NULL);
   const bool submit_only =
      G17P_COMPOSITE_GENERATE_MIPS &&
      getenv("G17P_GENERATED_MIPS_SUBMIT_ONLY") != NULL;
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());

   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("libvulkan.so.1 has no vkGetInstanceProcAddr");
   LOAD_GLOBAL(vkCreateInstance);

   const VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = PROBE_NAME,
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion =
         G17P_SHADER_QUAD_CONTROL ? VK_API_VERSION_1_3 :
         (G17P_INDIRECT_DRAW || G17P_LAYER_OUTPUT || G17P_MULTI_VIEWPORT)
                       ? VK_API_VERSION_1_2
                       : VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
#if G17P_F16_IO || G17P_INDIRECT_DRAW || G17P_BRESENHAM_LINES ||          \
   G17P_MULTIVIEW || G17P_LAYER_OUTPUT || G17P_DYNAMIC_BLEND ||           \
   G17P_ADVANCED_BLEND || G17P_MULTI_VIEWPORT ||                           \
   G17P_TEXTURE_ALPHA_COMPOSITE || G17P_SAMPLE_RATE_SHADING ||            \
   G17P_MSAA_COVERAGE || G17P_SHADER_QUAD_CONTROL || G17P_YCBCR ||        \
   G17P_DYNAMIC_VERTEX_INPUT || G17P_TEXTURE_MIN_LOD
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFeatures2);
#endif

#if G17P_INDIRECT_DRAW || G17P_SHADER_QUAD_CONTROL || G17P_DYNAMIC_VERTEX_INPUT
   LOAD_INSTANCE(instance, vkEnumerateDeviceExtensionProperties);
#endif
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFormatProperties);
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t physical_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   if (!physical_count)
      die("Vulkan loader found no physical device");

   VkPhysicalDevice *physical_devices =
      calloc(physical_count, sizeof(*physical_devices));
   if (!physical_devices)
      die("out of host memory");
   check(vkEnumeratePhysicalDevices(instance, &physical_count,
                                    physical_devices),
         "vkEnumeratePhysicalDevices(list)");

   VkPhysicalDevice physical = VK_NULL_HANDLE;
   uint32_t queue_family = UINT32_MAX;
   VkPhysicalDeviceProperties properties = {0};
   for (uint32_t p = 0; p < physical_count && !physical; ++p) {
      VkPhysicalDeviceProperties candidate;
      vkGetPhysicalDeviceProperties(physical_devices[p], &candidate);
#if !G17P_TEXTURE_MIN_LOD
      /* The min-LOD oracle also serves older GPUs, selected by the ICD. */
      if (candidate.deviceID != T8140_DEVICE_ID)
         continue;
#endif

      uint32_t queue_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[p],
                                                &queue_count, NULL);
      VkQueueFamilyProperties *queues = calloc(queue_count, sizeof(*queues));
      if (!queues)
         die("out of host memory");
      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[p],
                                                &queue_count, queues);
      for (uint32_t q = 0; q < queue_count; ++q) {
         const VkQueueFlags required =
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT;
         if ((queues[q].queueFlags & required) == required) {
            physical = physical_devices[p];
            queue_family = q;
            properties = candidate;
            break;
         }
      }
      free(queues);
   }
   free(physical_devices);
   if (!physical)
      die("no T8140 device has a graphics-and-transfer queue");

#if G17P_VERTEX_OUTPUT_68
   if (properties.limits.maxVertexOutputComponents < 68 ||
       properties.limits.maxFragmentInputComponents < 68)
      die("T8140/G17P 68-component vertex-output candidate is not enabled");
#endif

#if G17P_DYNAMIC_BLEND
   VkPhysicalDeviceExtendedDynamicState3FeaturesEXT available_dynamic3 = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 available_dynamic = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &available_dynamic3,
   };
   vkGetPhysicalDeviceFeatures2(physical, &available_dynamic);
   if (!available_dynamic3.extendedDynamicState3ColorBlendEnable ||
       !available_dynamic3.extendedDynamicState3ColorBlendEquation ||
       !available_dynamic3.extendedDynamicState3ColorWriteMask)
      die("required extended dynamic blend state is not advertised");
#elif G17P_MSAA_COVERAGE
   VkPhysicalDeviceExtendedDynamicState3FeaturesEXT available_dynamic3 = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 available_dynamic = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &available_dynamic3,
   };
   vkGetPhysicalDeviceFeatures2(physical, &available_dynamic);
#if G17P_MSAA_COVERAGE == 1 || G17P_MSAA_COVERAGE == 3
   if (!available_dynamic3.extendedDynamicState3SampleMask)
      die("dynamic sample-mask state is not advertised");
#else
   if (!available_dynamic3.extendedDynamicState3AlphaToCoverageEnable)
      die("dynamic alpha-to-coverage state is not advertised");
#endif
#endif

#if G17P_ADVANCED_BLEND
   VkPhysicalDeviceBlendOperationAdvancedFeaturesEXT available_advanced = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 available_advanced_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &available_advanced,
   };
   vkGetPhysicalDeviceFeatures2(physical, &available_advanced_features);
   if (!available_advanced.advancedBlendCoherentOperations)
      die("coherent advanced blend operations are not advertised");
#endif

#if G17P_TEXTURE_ALPHA_COMPOSITE
   VkPhysicalDeviceFeatures2 composite_available = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
   };
   vkGetPhysicalDeviceFeatures2(physical, &composite_available);
   if (!composite_available.features.shaderSampledImageArrayDynamicIndexing)
      die("dynamic sampled-image array indexing is not advertised");
#endif

#if G17P_F16_IO || G17P_MULTIVIEW
   VkPhysicalDeviceVulkan11Features available11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
   };
   VkPhysicalDeviceFeatures2 available = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &available11,
   };
   vkGetPhysicalDeviceFeatures2(physical, &available);
#if G17P_F16_IO
   if (!available11.storageInputOutput16)
      die("T8140/G17P does not advertise storageInputOutput16");
#endif
#if G17P_MULTIVIEW
   if (!available11.multiview || properties.limits.maxFramebufferLayers < 2)
      die("T8140/G17P does not advertise two-layer multiview");
#endif
#endif

#if G17P_BRESENHAM_LINES
   VkPhysicalDeviceLineRasterizationFeaturesKHR available_lines = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_KHR,
   };
   VkPhysicalDeviceFeatures2 available = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &available_lines,
   };
   vkGetPhysicalDeviceFeatures2(physical, &available);
   if (!available.features.wideLines || !available_lines.bresenhamLines)
      die("T8140/G17P lacks wideLines or bresenhamLines");
#endif

#if G17P_LAYER_OUTPUT
   VkPhysicalDeviceVulkan12Features layer_available12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
   };
   VkPhysicalDeviceFeatures2 layer_available = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &layer_available12,
   };
   vkGetPhysicalDeviceFeatures2(physical, &layer_available);
   if (!layer_available12.shaderOutputLayer ||
       properties.limits.maxFramebufferLayers < 2)
      die("T8140/G17P does not advertise vertex shader layer output");
#endif

#if G17P_MULTI_VIEWPORT
   VkPhysicalDeviceVulkan12Features viewport_available12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
   };
   VkPhysicalDeviceFeatures2 viewport_available = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &viewport_available12,
   };
   vkGetPhysicalDeviceFeatures2(physical, &viewport_available);
   if (!viewport_available.features.multiViewport ||
       !viewport_available12.shaderOutputViewportIndex ||
       properties.limits.maxViewports < 4)
      die("T8140/G17P does not advertise shader-selected multi-viewport");
#endif

#if G17P_SHADER_QUAD_CONTROL
   VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR maximal_available = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR,
   };
   VkPhysicalDeviceShaderQuadControlFeaturesKHR quad_available = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR,
      .pNext = &maximal_available,
   };
   VkPhysicalDeviceFeatures2 quad_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &quad_available,
   };
   vkGetPhysicalDeviceFeatures2(physical, &quad_features);
   if (!quad_available.shaderQuadControl ||
       !maximal_available.shaderMaximalReconvergence ||
       !has_device_extension(physical,
                             VK_KHR_SHADER_QUAD_CONTROL_EXTENSION_NAME) ||
       !has_device_extension(
          physical, VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME))
      die("T8140/G17P does not advertise divergent shader quad control");
#endif

#if G17P_INDIRECT_DRAW
   VkPhysicalDeviceVulkan12Features indirect_available12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
   };
   VkPhysicalDeviceFeatures2 indirect_available = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &indirect_available12,
   };
   vkGetPhysicalDeviceFeatures2(physical, &indirect_available);
   if (!indirect_available.features.drawIndirectFirstInstance ||
       !indirect_available.features.multiDrawIndirect)
      die("required core indirect-draw features are not advertised");
#if G17P_ROBUST_INDEXED
   if (!indirect_available.features.robustBufferAccess)
      die("robustBufferAccess is not advertised");
#endif
   if (!indirect_available12.drawIndirectCount ||
       !has_device_extension(physical,
                             VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME))
      die("qualified indirect-count support is not advertised");
#endif

#if G17P_SAMPLE_RATE_SHADING || G17P_MSAA_COVERAGE == 1
   VkPhysicalDeviceFeatures2 sample_rate_available = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
   };
   vkGetPhysicalDeviceFeatures2(physical, &sample_rate_available);
   if (!sample_rate_available.features.sampleRateShading)
      die("T8140/G17P does not advertise per-sample fragment shading");
#endif

   if (!(properties.limits.framebufferColorSampleCounts &
         (VkSampleCountFlags)G17P_MSAA_SAMPLES))
      die("requested color sample count is not advertised");

   VkFormatProperties format_properties;
   vkGetPhysicalDeviceFormatProperties(physical, IMAGE_FORMAT,
                                       &format_properties);
   const VkFormatFeatureFlags required_format =
      VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
   if ((format_properties.optimalTilingFeatures & required_format) !=
       required_format)
      die("target optimal images lack color-attachment or transfer-src support");

#if G17P_TEXTURE_ORACLE
   VkFormatProperties texture_format_properties;
   vkGetPhysicalDeviceFormatProperties(physical, TEXTURE_FORMAT,
                                       &texture_format_properties);
   const VkFormatFeatureFlags required_texture_format =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   if ((texture_format_properties.optimalTilingFeatures &
        required_texture_format) != required_texture_format)
      die("texture optimal images lack sampled/upload support");
#endif

#if G17P_DYNAMIC_VERTEX_INPUT
   VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT dynamic_vi_available = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 dynamic_vi_features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &dynamic_vi_available,
   };
   vkGetPhysicalDeviceFeatures2(physical, &dynamic_vi_features2);
   if (!dynamic_vi_available.vertexInputDynamicState)
      die("vertexInputDynamicState is not advertised");
   if (!has_device_extension(physical,
                             VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME))
      die("VK_EXT_vertex_input_dynamic_state is not advertised");
#endif

#if G17P_VERTEX_FORMAT
   VkFormatProperties vertex_format_properties;
   vkGetPhysicalDeviceFormatProperties(physical, VERTEX_INPUT_FORMAT,
                                       &vertex_format_properties);
   if (!(vertex_format_properties.bufferFeatures &
         VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT))
      die("required vertex-buffer format is not advertised");
#endif

   fprintf(stderr, PROBE_NAME ": device %s, queue %u\n",
           properties.deviceName, queue_family);

   const float queue_priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &queue_priority,
   };
#if G17P_HOST_IMAGE_COPY || G17P_TEXTURE_3D || G17P_BRESENHAM_LINES
#if G17P_HOST_IMAGE_COPY || G17P_TEXTURE_3D
   const VkPhysicalDeviceHostImageCopyFeaturesEXT host_copy_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT,
      .hostImageCopy = VK_TRUE,
   };
   const char *device_extensions[] = {
      VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME,
   };
#else
   const VkPhysicalDeviceLineRasterizationFeaturesKHR enabled_lines = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_KHR,
      .bresenhamLines = VK_TRUE,
   };
   const VkPhysicalDeviceFeatures enabled_features = {
      .wideLines = VK_TRUE,
   };
   const char *device_extensions[] = {
      VK_KHR_LINE_RASTERIZATION_EXTENSION_NAME,
   };
#endif
#endif
#if G17P_F16_IO || G17P_MULTIVIEW || G17P_YCBCR
   const VkPhysicalDeviceVulkan11Features enabled11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
#if G17P_F16_IO
      .storageInputOutput16 = VK_TRUE,
#endif
#if G17P_MULTIVIEW
      .multiview = VK_TRUE,
#endif
#if G17P_YCBCR
      .samplerYcbcrConversion = VK_TRUE,
#endif
   };
#endif
#if G17P_LAYER_OUTPUT
   const VkPhysicalDeviceVulkan12Features layer_enabled12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .shaderOutputLayer = VK_TRUE,
   };
#endif
#if G17P_MULTI_VIEWPORT
   const VkPhysicalDeviceVulkan12Features viewport_enabled12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .shaderOutputViewportIndex = VK_TRUE,
   };
   const VkPhysicalDeviceFeatures viewport_enabled = {
      .multiViewport = VK_TRUE,
   };
#endif
#if G17P_INDIRECT_DRAW
   const VkPhysicalDeviceVulkan12Features indirect_enabled12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .drawIndirectCount = G17P_INDIRECT_COUNT,
   };
   const VkPhysicalDeviceFeatures indirect_enabled = {
      .robustBufferAccess = G17P_ROBUST_INDEXED,
      .multiDrawIndirect = VK_TRUE,
      .drawIndirectFirstInstance = VK_TRUE,
   };
#endif
#if G17P_LOGIC_OP_XOR
   const VkPhysicalDeviceFeatures blend_features = {
      .logicOp = G17P_LOGIC_OP_XOR,
   };
#endif
#if G17P_TEXTURE_ALPHA_COMPOSITE
   const VkPhysicalDeviceFeatures composite_features = {
      .shaderSampledImageArrayDynamicIndexing = VK_TRUE,
   };
#endif
#if G17P_SAMPLE_RATE_SHADING || G17P_MSAA_COVERAGE == 1
   const VkPhysicalDeviceFeatures sample_rate_features = {
      .sampleRateShading = VK_TRUE,
   };
#endif
#if G17P_DYNAMIC_BLEND
   const VkPhysicalDeviceExtendedDynamicState3FeaturesEXT enabled_dynamic3 = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
      .extendedDynamicState3ColorBlendEnable = VK_TRUE,
      .extendedDynamicState3ColorBlendEquation = VK_TRUE,
      .extendedDynamicState3ColorWriteMask = VK_TRUE,
   };
   const char *dynamic_device_extensions[] = {
      VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
   };
#elif G17P_MSAA_COVERAGE
   const VkPhysicalDeviceExtendedDynamicState3FeaturesEXT enabled_dynamic3 = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
      .extendedDynamicState3SampleMask =
         G17P_MSAA_COVERAGE == 1 || G17P_MSAA_COVERAGE == 3,
      .extendedDynamicState3AlphaToCoverageEnable =
         G17P_MSAA_COVERAGE == 2,
   };
   const char *dynamic_device_extensions[] = {
      VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
   };
#endif
#if G17P_ADVANCED_BLEND
   const VkPhysicalDeviceBlendOperationAdvancedFeaturesEXT enabled_advanced = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_FEATURES_EXT,
      .advancedBlendCoherentOperations = VK_TRUE,
   };
   const char *advanced_device_extensions[] = {
      VK_EXT_BLEND_OPERATION_ADVANCED_EXTENSION_NAME,
   };
#endif
#if G17P_DYNAMIC_VERTEX_INPUT
   VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT dynamic_vi_enabled = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT,
      .vertexInputDynamicState = VK_TRUE,
   };
   const char *dynamic_vi_device_extensions[] = {
      VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME,
   };
#endif
#if G17P_SHADER_QUAD_CONTROL
   VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR maximal_enabled = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR,
      .shaderMaximalReconvergence = VK_TRUE,
   };
   VkPhysicalDeviceShaderQuadControlFeaturesKHR quad_enabled = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR,
      .pNext = &maximal_enabled,
      .shaderQuadControl = VK_TRUE,
   };
   const char *quad_device_extensions[] = {
      VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME,
      VK_KHR_SHADER_QUAD_CONTROL_EXTENSION_NAME,
   };
#endif
#if G17P_TEXTURE_MIN_LOD
   const char *clamp = getenv("G17P_VIEW_MIN_LOD");
#if G17P_TEXTURE_MIN_LOD >= 5
   if (clamp && strcmp(clamp, "0") && strcmp(clamp, "0.5") &&
       strcmp(clamp, "1") && strcmp(clamp, "1.5") && strcmp(clamp, "2"))
      die("G17P_VIEW_MIN_LOD must be 0, 0.5, 1, 1.5, or 2");
   view_min_lod = clamp ? strtof(clamp, NULL) : 2.0f;
#else
   if (clamp && strcmp(clamp, "0") && strcmp(clamp, "2"))
      die("G17P_VIEW_MIN_LOD must be 0 or 2");
   view_min_lod = clamp && !strcmp(clamp, "0") ? 0.0f : 2.0f;
#endif
   VkPhysicalDeviceImageViewMinLodFeaturesEXT min_lod_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 min_lod_available = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &min_lod_features,
   };
   vkGetPhysicalDeviceFeatures2(physical, &min_lod_available);
   if (!min_lod_features.minLod)
      die("imageViewMinLod is not advertised");
   const char *min_lod_extensions[] = {VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME};
#endif
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
#if G17P_TEXTURE_MIN_LOD
      .pNext = &min_lod_features,
#elif G17P_DYNAMIC_VERTEX_INPUT
      .pNext = &dynamic_vi_enabled,
#elif G17P_SHADER_QUAD_CONTROL
      .pNext = &quad_enabled,
#elif G17P_F16_IO || G17P_MULTIVIEW || G17P_YCBCR
      .pNext = &enabled11,
#elif G17P_BRESENHAM_LINES
      .pNext = &enabled_lines,
#elif G17P_LAYER_OUTPUT
      .pNext = &layer_enabled12,
#elif G17P_MULTI_VIEWPORT
      .pNext = &viewport_enabled12,
#elif G17P_HOST_IMAGE_COPY || G17P_TEXTURE_3D
      .pNext = &host_copy_features,
#elif G17P_INDIRECT_DRAW
      .pNext = &indirect_enabled12,
#elif G17P_DYNAMIC_BLEND
      .pNext = &enabled_dynamic3,
#elif G17P_MSAA_COVERAGE
      .pNext = &enabled_dynamic3,
#elif G17P_ADVANCED_BLEND
      .pNext = &enabled_advanced,
#endif
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
#if G17P_SAMPLE_RATE_SHADING || G17P_MSAA_COVERAGE == 1
      .pEnabledFeatures = &sample_rate_features,
#elif G17P_INDIRECT_DRAW
      .pEnabledFeatures = &indirect_enabled,
#elif G17P_MULTI_VIEWPORT
      .pEnabledFeatures = &viewport_enabled,
#elif G17P_TEXTURE_ALPHA_COMPOSITE
      .pEnabledFeatures = &composite_features,
#endif
#if G17P_BRESENHAM_LINES
      .pEnabledFeatures = &enabled_features,
#elif G17P_LOGIC_OP_XOR
      .pEnabledFeatures = &blend_features,
#endif
#if G17P_TEXTURE_MIN_LOD
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = min_lod_extensions,
#elif G17P_DYNAMIC_VERTEX_INPUT
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = dynamic_vi_device_extensions,
#elif G17P_HOST_IMAGE_COPY || G17P_TEXTURE_3D || G17P_BRESENHAM_LINES
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = device_extensions,
#elif G17P_DYNAMIC_BLEND
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = dynamic_device_extensions,
#elif G17P_MSAA_COVERAGE
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = dynamic_device_extensions,
#elif G17P_ADVANCED_BLEND
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = advanced_device_extensions,
#elif G17P_SHADER_QUAD_CONTROL
      .enabledExtensionCount = 2,
      .ppEnabledExtensionNames = quad_device_extensions,
#endif
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkGetDeviceQueue);
   LOAD_DEVICE(device, vkCreateBuffer);
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements);
   LOAD_DEVICE(device, vkCreateImage);
   LOAD_DEVICE(device, vkGetImageMemoryRequirements);
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkBindBufferMemory);
   LOAD_DEVICE(device, vkBindImageMemory);
   LOAD_DEVICE(device, vkMapMemory);
   LOAD_DEVICE(device, vkCreateImageView);
#if G17P_TEXTURE_ORACLE
   LOAD_DEVICE(device, vkCreateSampler);
#if G17P_YCBCR
   LOAD_DEVICE(device, vkCreateSamplerYcbcrConversion);
#endif
   LOAD_DEVICE(device, vkCreateDescriptorSetLayout);
   LOAD_DEVICE(device, vkCreateDescriptorPool);
   LOAD_DEVICE(device, vkAllocateDescriptorSets);
   LOAD_DEVICE(device, vkUpdateDescriptorSets);
#endif
   LOAD_DEVICE(device, vkCreateRenderPass);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkCreateGraphicsPipelines);
   LOAD_DEVICE(device, vkCreateFramebuffer);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkCmdBeginRenderPass);
#if G17P_SECONDARY_COMMAND_BUFFER
   LOAD_DEVICE(device, vkCmdExecuteCommands);
#endif
#if G17P_SECONDARY_COMMAND_BUFFER_MULTI
   LOAD_DEVICE(device, vkCmdPushConstants);
#endif
#if G17P_TEXTURE_ORACLE || G17P_RGBA8_MRT
   LOAD_DEVICE(device, vkCmdPipelineBarrier);
#endif
#if G17P_TEXTURE_ORACLE
#if G17P_SHADOW_COMPARE
   LOAD_DEVICE(device, vkCmdClearDepthStencilImage);
#else
   LOAD_DEVICE(device, vkCmdCopyBufferToImage);
#endif
#if G17P_COMPOSITE_GENERATE_MIPS
   LOAD_DEVICE(device, vkCmdBlitImage);
#endif
   LOAD_DEVICE(device, vkCmdBindDescriptorSets);
#endif
   LOAD_DEVICE(device, vkCmdBindPipeline);
   LOAD_DEVICE(device, vkCmdBindVertexBuffers);
   LOAD_DEVICE(device, vkCmdDraw);
#if G17P_DYNAMIC_BLEND || G17P_TEXTURE_ALPHA_COMPOSITE ||                 \
   G17P_MULTI_VIEWPORT_DYNAMIC_SCISSOR || G17P_MSAA_COVERAGE == 2 ||     \
   G17P_MSAA_COVERAGE == 3
   LOAD_DEVICE(device, vkCmdSetScissor);
#endif
#if G17P_MULTI_VIEWPORT_DYNAMIC_VIEWPORT
   LOAD_DEVICE(device, vkCmdSetViewport);
#endif
#if G17P_DYNAMIC_BLEND
   LOAD_DEVICE(device, vkCmdSetColorBlendEnableEXT);
   LOAD_DEVICE(device, vkCmdSetColorBlendEquationEXT);
   LOAD_DEVICE(device, vkCmdSetColorWriteMaskEXT);
#endif
#if G17P_DYNAMIC_VERTEX_INPUT
   LOAD_DEVICE(device, vkCmdSetVertexInputEXT);
#endif
#if G17P_MSAA_COVERAGE == 1 || G17P_MSAA_COVERAGE == 3
   LOAD_DEVICE(device, vkCmdSetSampleMaskEXT);
#elif G17P_MSAA_COVERAGE == 2
   LOAD_DEVICE(device, vkCmdSetAlphaToCoverageEnableEXT);
#endif
#if G17P_INDIRECT_DRAW
   LOAD_DEVICE(device, vkCmdDrawIndirect);
   LOAD_DEVICE(device, vkCmdDrawIndexedIndirect);
#if G17P_INDIRECT_COUNT
   LOAD_DEVICE(device, vkCmdDrawIndirectCount);
   LOAD_DEVICE(device, vkCmdDrawIndexedIndirectCount);
#endif
#endif
#if G17P_INDIRECT_DRAW || G17P_INDEX_RESTART || \
   G17P_TEXTURE_ALPHA_COMPOSITE
   LOAD_DEVICE(device, vkCmdBindIndexBuffer);
#endif
#if G17P_INDEX_RESTART || G17P_TEXTURE_ALPHA_COMPOSITE
   LOAD_DEVICE(device, vkCmdDrawIndexed);
#endif
   LOAD_DEVICE(device, vkCmdEndRenderPass);
#if G17P_HOST_IMAGE_COPY
   LOAD_DEVICE(device, vkCopyImageToMemoryEXT);
#else
   LOAD_DEVICE(device, vkCmdCopyImageToBuffer);
#endif
#if G17P_TEXTURE_3D
   LOAD_DEVICE(device, vkCopyMemoryToImageEXT);
   LOAD_DEVICE(device, vkTransitionImageLayoutEXT);
#endif
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);

   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);

   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

#if G17P_VERTEX_FORMAT
   enum {
      VERTEX_FORMAT_BINDING = 3,
      VERTEX_FORMAT_BIND_OFFSET = 64,
#if G17P_VERTEX_FORMAT == 1
      VERTEX_FORMAT_STRIDE = 13,
      VERTEX_FORMAT_ATTRIBUTE_OFFSET = 5,
#elif G17P_VERTEX_FORMAT == 2
      VERTEX_FORMAT_STRIDE = 14,
      VERTEX_FORMAT_ATTRIBUTE_OFFSET = 6,
#else
      VERTEX_FORMAT_STRIDE = 10,
      VERTEX_FORMAT_ATTRIBUTE_OFFSET = 6,
#endif
      VERTEX_FORMAT_COUNT = 3,
      VERTEX_FORMAT_SUFFIX_BYTES = 64,
      VERTEX_FORMAT_BUFFER_BYTES = VERTEX_FORMAT_BIND_OFFSET +
                                   VERTEX_FORMAT_COUNT *
                                      VERTEX_FORMAT_STRIDE +
                                   VERTEX_FORMAT_SUFFIX_BYTES,
   };
   uint8_t vertex_format_expected[VERTEX_FORMAT_BUFFER_BYTES];
   memset(vertex_format_expected, 0xa5, sizeof(vertex_format_expected));
#if G17P_VERTEX_FORMAT == 1
   static const uint8_t packed_positions[VERTEX_FORMAT_COUNT][4] = {
      {0, 0, 0x35, 0xc7},
      {255, 0, 0x6a, 0x91},
      {0, 255, 0xbd, 0x4e},
   };
#elif G17P_VERTEX_FORMAT == 2
   static const int16_t packed_positions[VERTEX_FORMAT_COUNT][2] = {
      {-1, -1},
      {3, -1},
      {-1, 3},
   };
#else
   static const uint16_t packed_positions[VERTEX_FORMAT_COUNT][2] = {
      {0xbc00, 0xbc00},
      {0x4200, 0xbc00},
      {0xbc00, 0x4200},
   };
#endif
   for (uint32_t i = 0; i < VERTEX_FORMAT_COUNT; ++i) {
      uint8_t *record = vertex_format_expected + VERTEX_FORMAT_BIND_OFFSET +
                        i * VERTEX_FORMAT_STRIDE;
      memset(record, 0, VERTEX_FORMAT_STRIDE);
      memcpy(record + VERTEX_FORMAT_ATTRIBUTE_OFFSET, packed_positions[i],
             sizeof(packed_positions[i]));
   }
   struct mapped_buffer vertex_format = create_mapped_buffer(
      device, &memory_properties, sizeof(vertex_format_expected),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, "packed vertex-format buffer");
   memcpy(vertex_format.map, vertex_format_expected,
          sizeof(vertex_format_expected));
#elif G17P_INSTANCE_RATE
   enum {
      INSTANCE_VERTEX_BIND_OFFSET = 64,
      INSTANCE_VERTEX_STRIDE = 24,
      INSTANCE_VERTEX_ATTRIBUTE_OFFSET = 8,
      INSTANCE_VERTEX_COUNT = 6,
      INSTANCE_VERTEX_SUFFIX_BYTES = 64,
      INSTANCE_VERTEX_BUFFER_BYTES = INSTANCE_VERTEX_BIND_OFFSET +
                                     INSTANCE_VERTEX_COUNT *
                                        INSTANCE_VERTEX_STRIDE +
                                     INSTANCE_VERTEX_SUFFIX_BYTES,
      INSTANCE_DATA_BIND_OFFSET = 96,
      INSTANCE_DATA_STRIDE = 32,
      INSTANCE_DATA_TRANSFORM_OFFSET = 8,
      INSTANCE_DATA_COUNT = 6,
      INSTANCE_DATA_SUFFIX_BYTES = 64,
      INSTANCE_DATA_BUFFER_BYTES = INSTANCE_DATA_BIND_OFFSET +
                                   INSTANCE_DATA_COUNT * INSTANCE_DATA_STRIDE +
                                   INSTANCE_DATA_SUFFIX_BYTES,
   };
   static const float positions[INSTANCE_VERTEX_COUNT][2] = {
      {-1.0f, -1.0f}, {1.0f, -1.0f}, {-1.0f, 1.0f},
      {-1.0f, 1.0f},  {1.0f, -1.0f}, {1.0f, 1.0f},
   };
   uint8_t vertex_expected[INSTANCE_VERTEX_BUFFER_BYTES];
   memset(vertex_expected, 0xa5, sizeof(vertex_expected));
   for (uint32_t i = 0; i < INSTANCE_VERTEX_COUNT; ++i) {
      uint8_t *record = vertex_expected + INSTANCE_VERTEX_BIND_OFFSET +
                        i * INSTANCE_VERTEX_STRIDE;
      memset(record, 0, INSTANCE_VERTEX_STRIDE);
      memcpy(record + INSTANCE_VERTEX_ATTRIBUTE_OFFSET, positions[i],
             sizeof(positions[i]));
   }
   struct mapped_buffer vertex = create_mapped_buffer(
      device, &memory_properties, sizeof(vertex_expected),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, "instance-rate vertex buffer");
   memcpy(vertex.map, vertex_expected, sizeof(vertex_expected));

   uint8_t instance_expected[INSTANCE_DATA_BUFFER_BYTES];
   memset(instance_expected, 0xa5, sizeof(instance_expected));
   memset(instance_expected + INSTANCE_DATA_BIND_OFFSET, 0,
          INSTANCE_DATA_COUNT * INSTANCE_DATA_STRIDE);
   static const float transforms[2][4] = {
      {0.5f, 1.0f, -0.5f, 0.0f},
      {0.5f, 1.0f, 0.5f, 0.0f},
   };
   for (uint32_t i = 0; i < 2; ++i) {
      uint8_t *record = instance_expected + INSTANCE_DATA_BIND_OFFSET +
                        (i + 3) * INSTANCE_DATA_STRIDE;
      memcpy(record + INSTANCE_DATA_TRANSFORM_OFFSET, transforms[i],
             sizeof(transforms[i]));
   }
   struct mapped_buffer instance_data = create_mapped_buffer(
      device, &memory_properties, sizeof(instance_expected),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, "instance-rate data buffer");
   memcpy(instance_data.map, instance_expected, sizeof(instance_expected));
#elif G17P_INDEX_RESTART
   enum {
      RESTART_VERTEX_COUNT = UINT16_MAX + 1u,
      RESTART_VERTEX_OFFSET = 64,
      RESTART_VERTEX_DATA_BYTES = RESTART_VERTEX_COUNT * 2 * sizeof(float),
      RESTART_VERTEX_SUFFIX_BYTES = 64,
      RESTART_VERTEX_BUFFER_BYTES = RESTART_VERTEX_OFFSET +
                                    RESTART_VERTEX_DATA_BYTES +
                                    RESTART_VERTEX_SUFFIX_BYTES,
   };
   uint8_t *vertex_expected = malloc(RESTART_VERTEX_BUFFER_BYTES);
   if (!vertex_expected)
      die("out of host memory");
   memset(vertex_expected, 0xa5, RESTART_VERTEX_BUFFER_BYTES);
   float (*restart_vertices)[2] =
      (void *)(vertex_expected + RESTART_VERTEX_OFFSET);
   for (uint32_t i = 0; i < RESTART_VERTEX_COUNT; ++i) {
      restart_vertices[i][0] = 2.0f;
      restart_vertices[i][1] = 2.0f;
   }
   restart_vertices[3][0] = -1.0f;
   restart_vertices[3][1] = -1.0f;
   restart_vertices[4][0] = 3.0f;
   restart_vertices[4][1] = -1.0f;
   restart_vertices[5][0] = -1.0f;
   restart_vertices[5][1] = 3.0f;
   restart_vertices[6][0] = 3.0f;
   restart_vertices[6][1] = 3.0f;
   restart_vertices[UINT16_MAX][0] = 3.0f;
   restart_vertices[UINT16_MAX][1] = 3.0f;

   struct mapped_buffer vertex = create_mapped_buffer(
      device, &memory_properties, RESTART_VERTEX_BUFFER_BYTES,
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, "restart vertex buffer");
   memcpy(vertex.map, vertex_expected, RESTART_VERTEX_BUFFER_BYTES);
   VkBuffer vertex_buffer = vertex.buffer;
#elif G17P_TEXTURE_ORACLE
   struct composite_vertex {
      float position[2];
      uint32_t source_bias;
   };
   enum {
      TEXTURE_VERTEX_OFFSET = 64,
      TEXTURE_VERTEX_SUFFIX_BYTES = 64,
      TEXTURE_VERTEX_BYTES = TEXTURE_VERTEX_OFFSET +
                             (G17P_TEXTURE_ALPHA_COMPOSITE
                                 ? 4 * sizeof(struct composite_vertex)
                                 : 3 * 2 * sizeof(float)) +
                             TEXTURE_VERTEX_SUFFIX_BYTES,
   };
#if G17P_TEXTURE_ALPHA_COMPOSITE
   /* Two triangles sharing a full-surface diagonal. Every pixel, including
    * the shared edge, must be covered exactly once. The integer attribute
    * forces a flat input ahead of the smooth UV in the fragment program. */
#if G17P_COMPOSITE_FULLSCREEN_TRIANGLE
   /* One oversized triangle removes the internal quad seam while retaining
    * exactly the same UV gradient and fragment program. */
   static const struct composite_vertex vertices[] = {
      {{-1.0f, -1.0f}, 1},
      {{3.0f, -1.0f}, 1},
      {{-1.0f, 3.0f}, 1},
   };
#elif G17P_COMPOSITE_FRACTIONAL_INSET
   /* A 10x9 quad translated by quarter pixels. Its clip coordinates are
    * fractional for this odd target, while sample centres 4..13 by 3..11
    * remain safely inside its outer edges. */
   static const struct composite_vertex vertices[] = {
      {{(2.0f * 4.25f / IMAGE_WIDTH) - 1.0f,
        (2.0f * 3.25f / IMAGE_HEIGHT) - 1.0f}, 1},
      {{(2.0f * 14.25f / IMAGE_WIDTH) - 1.0f,
        (2.0f * 3.25f / IMAGE_HEIGHT) - 1.0f}, 1},
      {{(2.0f * 4.25f / IMAGE_WIDTH) - 1.0f,
        (2.0f * 12.25f / IMAGE_HEIGHT) - 1.0f}, 1},
      {{(2.0f * 14.25f / IMAGE_WIDTH) - 1.0f,
        (2.0f * 12.25f / IMAGE_HEIGHT) - 1.0f}, 1},
   };
#else
   static const struct composite_vertex vertices[] = {
      {{-1.0f, -1.0f}, 1},
      {{1.0f, -1.0f}, 1},
      {{-1.0f, 1.0f}, 1},
      {{1.0f, 1.0f}, 1},
   };
#endif
#else
   static const float vertices[][2] = {
      {-1.0f, -1.0f},
      {3.0f, -1.0f},
      {-1.0f, 3.0f},
   };
#endif
   uint8_t texture_vertex_expected[TEXTURE_VERTEX_BYTES];
   memset(texture_vertex_expected, 0xa5, sizeof(texture_vertex_expected));
   memcpy(texture_vertex_expected + TEXTURE_VERTEX_OFFSET, vertices,
          sizeof(vertices));
   struct mapped_buffer texture_vertex = create_mapped_buffer(
      device, &memory_properties, sizeof(texture_vertex_expected),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, "texture-oracle vertex buffer");
   memcpy(texture_vertex.map, texture_vertex_expected,
          sizeof(texture_vertex_expected));
   VkBuffer vertex_buffer = texture_vertex.buffer;
#else
#if G17P_BRESENHAM_LINES
   /* A horizontal line centred on row 16.5 and clipped past both viewport
    * sides. Width 3.0 must cover exactly rows 15..17, all 32 columns. */
   static const float vertices[][2] = {
      {-4.0f, 0.03125f},
      {4.0f, 0.03125f},
   };
#elif G17P_INDIRECT_DRAW
   /* The first three vertices are deliberately degenerate.  A zero/wrong
    * firstVertex, firstIndex, or baseVertex therefore cannot paint the image. */
   static const float vertices[][2] = {
      {0.0f, 0.0f},
      {0.0f, 0.0f},
      {0.0f, 0.0f},
      {-1.0f, -1.0f},
      {3.0f, -1.0f},
      {-1.0f, 3.0f},
   };
#elif G17P_SECONDARY_COMMAND_BUFFER_MULTI
   /* Four disjoint clip-space quads, one per quadrant, six vertices each.
    * Each secondary selects its own quadrant through firstVertex alone, so
    * the pipeline, the render pass and the vertex buffer are shared and the
    * only new variable against the single-secondary oracle is how many
    * secondaries the pass replays. The quadrant edges land exactly on the
    * pixel-16 boundary, so the quads neither overlap nor leave a crack. */
   static const float vertices[][2] = {
      {-1.0f, -1.0f}, {0.0f, -1.0f}, {-1.0f, 0.0f},
      {0.0f, -1.0f}, {0.0f, 0.0f}, {-1.0f, 0.0f},
      {0.0f, -1.0f}, {1.0f, -1.0f}, {0.0f, 0.0f},
      {1.0f, -1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f},
      {-1.0f, 0.0f}, {0.0f, 0.0f}, {-1.0f, 1.0f},
      {0.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 1.0f},
      {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f},
      {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f},
   };
#else
   static const float vertices[][2] = {
      {-1.0f, -1.0f},
      {3.0f, -1.0f},
      {-1.0f, 3.0f},
   };
#endif
   const VkBufferCreateInfo vertex_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(vertices),
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer vertex_buffer;
   check(vkCreateBuffer(device, &vertex_info, NULL, &vertex_buffer),
         "vkCreateBuffer(vertex)");
   VkMemoryRequirements vertex_requirements;
   vkGetBufferMemoryRequirements(device, vertex_buffer, &vertex_requirements);
   const uint32_t vertex_memory_type = find_memory_type(
      &memory_properties, vertex_requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   const VkMemoryAllocateInfo vertex_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = vertex_requirements.size,
      .memoryTypeIndex = vertex_memory_type,
   };
   VkDeviceMemory vertex_memory;
   check(vkAllocateMemory(device, &vertex_memory_info, NULL, &vertex_memory),
         "vkAllocateMemory(vertex)");
   check(vkBindBufferMemory(device, vertex_buffer, vertex_memory, 0),
         "vkBindBufferMemory(vertex)");
   void *vertex_map = NULL;
   check(vkMapMemory(device, vertex_memory, 0, sizeof(vertices), 0,
                     &vertex_map),
         "vkMapMemory(vertex)");
   memcpy(vertex_map, vertices, sizeof(vertices));
#endif

#if G17P_INDIRECT_DRAW
   enum {
      INDIRECT_COMMAND_OFFSET = 64,
      INDIRECT_COMMAND_STRIDE = 32,
      INDIRECT_SUFFIX_BYTES = 64,
   };
#if !G17P_INDIRECT_INDEXED
   const VkDrawIndirectCommand indirect_command = {
      .vertexCount = 3,
      .instanceCount = 1,
      .firstVertex = 3,
      .firstInstance = 5,
   };
#else
   const VkDrawIndexedIndirectCommand indirect_command = {
      .indexCount = 3,
      .instanceCount = 1,
      .firstIndex = 3,
      .vertexOffset = 2,
      .firstInstance = 5,
   };
#endif
   uint8_t indirect_expected[INDIRECT_COMMAND_OFFSET +
                             2 * INDIRECT_COMMAND_STRIDE +
                             sizeof(indirect_command) +
                             INDIRECT_SUFFIX_BYTES];
   memset(indirect_expected, 0xa5, sizeof(indirect_expected));
   /* If the implementation loses the API offset, both records at the buffer
    * base are harmless zero draws. At the requested offset, the first record
    * is also zero and only the second, stride-selected record may paint. */
   memset(indirect_expected, 0, sizeof(indirect_command));
   memset(indirect_expected + INDIRECT_COMMAND_STRIDE, 0,
          sizeof(indirect_command));
   memset(indirect_expected + INDIRECT_COMMAND_OFFSET, 0,
          sizeof(indirect_command));
   memset(indirect_expected + INDIRECT_COMMAND_OFFSET +
             2 * INDIRECT_COMMAND_STRIDE,
          0, sizeof(indirect_command));
   memcpy(indirect_expected + INDIRECT_COMMAND_OFFSET +
             INDIRECT_COMMAND_STRIDE,
          &indirect_command, sizeof(indirect_command));
   struct mapped_buffer indirect = create_mapped_buffer(
      device, &memory_properties, sizeof(indirect_expected),
      VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, "indirect buffer");
   memcpy(indirect.map, indirect_expected, sizeof(indirect_expected));

#if G17P_INDIRECT_COUNT
   enum {
      INDIRECT_COUNT_OFFSET = 64,
      INDIRECT_COUNT_SUFFIX_BYTES = 64,
   };
   uint8_t indirect_count_expected[INDIRECT_COUNT_OFFSET + sizeof(uint32_t) +
                                   INDIRECT_COUNT_SUFFIX_BYTES];
   memset(indirect_count_expected, 0xa5, sizeof(indirect_count_expected));
   memset(indirect_count_expected, 0, sizeof(uint32_t));
   const uint32_t indirect_count_value = 2;
   memcpy(indirect_count_expected + INDIRECT_COUNT_OFFSET,
          &indirect_count_value, sizeof(indirect_count_value));
   struct mapped_buffer indirect_count = create_mapped_buffer(
      device, &memory_properties, sizeof(indirect_count_expected),
      VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, "indirect count buffer");
   memcpy(indirect_count.map, indirect_count_expected,
          sizeof(indirect_count_expected));
#endif

#if G17P_INDIRECT_INDEXED
   enum {
      INDEX_BUFFER_OFFSET = 64,
      INDEX_SUFFIX_BYTES = 64,
   };
   static const uint16_t indices[] = {0, 0, 0, 1, 2, 3};
   uint8_t index_expected[INDEX_BUFFER_OFFSET + sizeof(indices) +
                          INDEX_SUFFIX_BYTES];
   memset(index_expected, 0xa5, sizeof(index_expected));
   /* Ignoring either the binding offset or firstIndex selects only zero
    * indices, yielding a degenerate triangle instead of an out-of-range read. */
   memset(index_expected, 0, sizeof(indices));
   memcpy(index_expected + INDEX_BUFFER_OFFSET, indices, sizeof(indices));
   struct mapped_buffer index = create_mapped_buffer(
      device, &memory_properties, sizeof(index_expected),
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT, "index buffer");
   memcpy(index.map, index_expected, sizeof(index_expected));
#endif
#endif

#if G17P_INDEX_RESTART
   enum {
      RESTART_INDEX_BUFFER_OFFSET = 64,
      RESTART_FIRST_INDEX = 31,
      RESTART_INDEX_SEQUENCE_OFFSET = RESTART_INDEX_BUFFER_OFFSET +
                                      RESTART_FIRST_INDEX * sizeof(uint16_t),
      RESTART_INDEX_SUFFIX_BYTES = 64,
   };
   static const uint16_t restart_indices[] = {
      3, 4, 5, UINT16_MAX, 6, 6, 6,
   };
   uint8_t restart_index_expected[RESTART_INDEX_SEQUENCE_OFFSET +
                                  sizeof(restart_indices) +
                                  RESTART_INDEX_SUFFIX_BYTES];
   memset(restart_index_expected, 0xa5, sizeof(restart_index_expected));
   /* Ignoring either the binding offset or firstIndex selects only safe zero
    * indices. The requested seven-index sequence starts at byte 126 and
    * crosses the 128-byte boundary. */
   memset(restart_index_expected, 0, RESTART_INDEX_SEQUENCE_OFFSET);
   memcpy(restart_index_expected + RESTART_INDEX_SEQUENCE_OFFSET,
          restart_indices, sizeof(restart_indices));
   struct mapped_buffer restart_index = create_mapped_buffer(
      device, &memory_properties, sizeof(restart_index_expected),
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT, "restart index buffer");
   memcpy(restart_index.map, restart_index_expected,
          sizeof(restart_index_expected));
#endif

#if G17P_TEXTURE_ALPHA_COMPOSITE
   enum {
      COMPOSITE_INDEX_BUFFER_OFFSET = 64,
      COMPOSITE_FIRST_INDEX = 5,
      COMPOSITE_INDEX_SEQUENCE_OFFSET = COMPOSITE_INDEX_BUFFER_OFFSET +
                                        COMPOSITE_FIRST_INDEX *
                                           sizeof(uint16_t),
      COMPOSITE_INDEX_SUFFIX_BYTES = 64,
      COMPOSITE_UNIFORM_OFFSET = 64,
      COMPOSITE_UNIFORM_STRIDE = 64,
      COMPOSITE_OPACITY_COUNT = G17P_COMPOSITE_OPACITY_COUNT,
      COMPOSITE_UNIFORM_SUFFIX_BYTES = 64,
   };
#if G17P_COMPOSITE_SPLIT == 0
   /* First diagonal, counter-clockwise winding. */
   static const uint16_t composite_indices[] = {0, 1, 2, 2, 1, 3};
#elif G17P_COMPOSITE_SPLIT == 1
   /* Opposite diagonal, counter-clockwise winding. */
   static const uint16_t composite_indices[] = {0, 1, 3, 0, 3, 2};
#elif G17P_COMPOSITE_SPLIT == 2
   /* First diagonal, clockwise winding. */
   static const uint16_t composite_indices[] = {2, 1, 0, 3, 1, 2};
#else
   /* Opposite diagonal, clockwise winding. */
   static const uint16_t composite_indices[] = {3, 1, 0, 2, 3, 0};
#endif
   uint8_t composite_index_expected[COMPOSITE_INDEX_SEQUENCE_OFFSET +
                                    sizeof(composite_indices) +
                                    COMPOSITE_INDEX_SUFFIX_BYTES];
   memset(composite_index_expected, 0xa5, sizeof(composite_index_expected));
   /* Ignoring either the binding offset or firstIndex selects only zeroes,
    * producing a degenerate draw without making an out-of-range access. */
   memset(composite_index_expected, 0, COMPOSITE_INDEX_SEQUENCE_OFFSET);
   memcpy(composite_index_expected + COMPOSITE_INDEX_SEQUENCE_OFFSET,
          composite_indices, sizeof(composite_indices));
   struct mapped_buffer composite_index = create_mapped_buffer(
      device, &memory_properties, sizeof(composite_index_expected),
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT, "composite index buffer");
   memcpy(composite_index.map, composite_index_expected,
          sizeof(composite_index_expected));

   struct composite_params {
      uint32_t source_index;
      float opacity;
      float texel_offset[2];
   };
   uint8_t composite_uniform_expected[COMPOSITE_UNIFORM_OFFSET +
                                      COMPOSITE_OPACITY_COUNT *
                                         COMPOSITE_UNIFORM_STRIDE +
                                      COMPOSITE_UNIFORM_SUFFIX_BYTES];
   memset(composite_uniform_expected, 0xa5,
          sizeof(composite_uniform_expected));
   /* A lost descriptor offset safely selects source 0 with zero opacity. Each
    * aligned dynamic record selects source 1 and advances opacity by 1/16,
    * including the transparent and opaque endpoints. */
   memset(composite_uniform_expected, 0, COMPOSITE_UNIFORM_OFFSET);
   for (unsigned step = 0; step < COMPOSITE_OPACITY_COUNT; ++step) {
      const struct composite_params params = {
         .source_index = 1,
         .opacity = step / 16.0f,
         .texel_offset = {
            1.0f / (4.0f * IMAGE_WIDTH),
            1.0f / (4.0f * IMAGE_HEIGHT),
         },
      };
      memcpy(composite_uniform_expected + COMPOSITE_UNIFORM_OFFSET +
                step * COMPOSITE_UNIFORM_STRIDE,
             &params, sizeof(params));
   }
   struct mapped_buffer composite_uniform = create_mapped_buffer(
      device, &memory_properties, sizeof(composite_uniform_expected),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "composite uniform buffer");
   memcpy(composite_uniform.map, composite_uniform_expected,
          sizeof(composite_uniform_expected));
#endif

#if G17P_TEXTURE_ORACLE
   enum {
      TEXTURE_WIDTH = G17P_TEXTURE_ALPHA_COMPOSITE
                         ? IMAGE_WIDTH * 2 + G17P_COMPOSITE_GENERATE_MIPS_ODD
                         : 64,
      TEXTURE_HEIGHT = G17P_TEXTURE_ALPHA_COMPOSITE
                          ? IMAGE_HEIGHT * 2 + G17P_COMPOSITE_GENERATE_MIPS_ODD
                          : 64,
#if G17P_YCBCR
      TEXTURE_LEVELS = 1,
      TEXTURE_LAYERS = 1,
#elif G17P_CUBE_SAMPLE
      TEXTURE_LEVELS = 1,
      TEXTURE_LAYERS = 6,
#elif G17P_TEXTURE_3D
      TEXTURE_LEVELS = 2,
      TEXTURE_LAYERS = 1,
#elif G17P_TEXTURE_ALPHA_COMPOSITE
      TEXTURE_LEVELS = 3,
      TEXTURE_LAYERS = G17P_COMPOSITE_GENERATE_MIPS ? 1 : 2,
#elif G17P_TWO_SAMPLERS
      /* One 64x64 image supplies both sampled textures as distinct array
       * layers, so the two descriptors differ in nothing but their view. */
      TEXTURE_LEVELS = 3,
      TEXTURE_LAYERS = 2,
#else
      TEXTURE_LEVELS = 3,
      TEXTURE_LAYERS = 1,
#endif
      TEXTURE_DEPTH = G17P_TEXTURE_3D ? 4 : 1,
      TEXTURE_GUARD_BYTES = 64,
#if G17P_YCBCR
      TEXTURE_LEVEL0_BYTES = TEXTURE_WIDTH * TEXTURE_HEIGHT,
      TEXTURE_LEVEL1_BYTES = (TEXTURE_WIDTH / 2) * (TEXTURE_HEIGHT / 2) * 2,
      TEXTURE_LEVEL2_BYTES = 0,
#else
      TEXTURE_LEVEL0_BYTES =
         TEXTURE_WIDTH * TEXTURE_HEIGHT * TEXTURE_PIXEL_BYTES *
            TEXTURE_LAYERS * TEXTURE_DEPTH,
      TEXTURE_LEVEL1_BYTES = G17P_CUBE_SAMPLE
         ? 0
         : (TEXTURE_WIDTH / 2) * (TEXTURE_HEIGHT / 2) * TEXTURE_PIXEL_BYTES *
              (G17P_TEXTURE_3D ? 2 : TEXTURE_LAYERS),
      TEXTURE_LEVEL2_BYTES =
         (G17P_CUBE_SAMPLE || G17P_TEXTURE_3D)
            ? 0
            : (TEXTURE_WIDTH / 4) * (TEXTURE_HEIGHT / 4) *
                 TEXTURE_PIXEL_BYTES * TEXTURE_LAYERS,
#endif
      TEXTURE_DATA_BYTES = TEXTURE_LEVEL0_BYTES + TEXTURE_LEVEL1_BYTES +
                           TEXTURE_LEVEL2_BYTES,
      TEXTURE_UPLOAD_BYTES = TEXTURE_DATA_BYTES + 2 * TEXTURE_GUARD_BYTES,
   };
#if !G17P_SHADOW_COMPARE
   uint8_t texture_upload_expected[TEXTURE_UPLOAD_BYTES];
   memset(texture_upload_expected, 0xa5, sizeof(texture_upload_expected));
#if G17P_YCBCR
   /* Plane 0 is Y=51. Plane 1 stores interleaved Cb=102, Cr=204. The
    * conversion uses RGB_IDENTITY so the exact shader result is Cr,Y,Cb. */
   memset(texture_upload_expected + TEXTURE_GUARD_BYTES, 51,
          TEXTURE_LEVEL0_BYTES);
   uint8_t *chroma = texture_upload_expected + TEXTURE_GUARD_BYTES +
                     TEXTURE_LEVEL0_BYTES;
   for (size_t texel = 0; texel < (TEXTURE_WIDTH / 2) * (TEXTURE_HEIGHT / 2);
        ++texel) {
      chroma[texel * 2 + 0] = 102;
      chroma[texel * 2 + 1] = 204;
   }
#elif G17P_CUBE_SAMPLE
   static const uint8_t face_colours[TEXTURE_LAYERS][4] = {
      {0xff, 0x00, 0x00, 0xff}, {0x00, 0xff, 0x00, 0xff},
      {0x00, 0x00, 0xff, 0xff}, {0xff, 0xff, 0x00, 0xff},
      {0xff, 0x00, 0xff, 0xff}, {0x00, 0xff, 0xff, 0xff},
   };
   for (unsigned layer = 0; layer < TEXTURE_LAYERS; ++layer) {
      uint8_t *dst = texture_upload_expected + TEXTURE_GUARD_BYTES +
                     layer * TEXTURE_WIDTH * TEXTURE_HEIGHT *
                        TEXTURE_PIXEL_BYTES;
      for (size_t pixel = 0; pixel < TEXTURE_WIDTH * TEXTURE_HEIGHT;
           ++pixel)
         memcpy(dst + pixel * TEXTURE_PIXEL_BYTES, face_colours[layer],
                TEXTURE_PIXEL_BYTES);
   }
#elif G17P_TEXTURE_3D
   static const uint8_t slice_colours[TEXTURE_DEPTH][4] = {
      {0xff, 0x00, 0x00, 0xff}, {0x00, 0xff, 0x00, 0xff},
      {0x00, 0x00, 0xff, 0xff}, {0xff, 0xff, 0x00, 0xff},
   };
   for (unsigned slice = 0; slice < TEXTURE_DEPTH; ++slice) {
      uint8_t *dst = texture_upload_expected + TEXTURE_GUARD_BYTES +
                     slice * TEXTURE_WIDTH * TEXTURE_HEIGHT *
                        TEXTURE_PIXEL_BYTES;
      for (size_t pixel = 0; pixel < TEXTURE_WIDTH * TEXTURE_HEIGHT;
           ++pixel)
         memcpy(dst + pixel * TEXTURE_PIXEL_BYTES, slice_colours[slice],
                TEXTURE_PIXEL_BYTES);
   }
   static const uint8_t mip1_colours[2][4] = {
      {0xff, 0x00, 0x00, 0xff}, {0x00, 0x00, 0xff, 0xff},
   };
   uint8_t *mip1 = texture_upload_expected + TEXTURE_GUARD_BYTES +
                   TEXTURE_LEVEL0_BYTES;
   for (unsigned slice = 0; slice < 2; ++slice) {
      for (size_t pixel = 0; pixel < 32 * 32; ++pixel)
         memcpy(mip1 + (slice * 32 * 32 + pixel) * TEXTURE_PIXEL_BYTES,
                mip1_colours[slice], TEXTURE_PIXEL_BYTES);
   }
#elif G17P_TEXTURE_ALPHA_COMPOSITE
   /* Descriptor 0 is a yellow negative control at every level. Descriptor 1
    * is the red/checker/blue LOD ladder: a texture twice the framebuffer size
    * makes implicit derivatives select mip 1. Its transparent/green checker
    * is premultiplied already, so linear interpolation remains premultiplied. */
#if G17P_COMPOSITE_GENERATE_MIPS
   /* GTK uploads a single-layer premultiplied BGRA8 base level, then builds
    * the complete chain with same-image linear blits. Each 2x2 base-level
    * block is uniform, so generated mip 1 is the exact transparent/green
    * checker expected by the compositor verifier. */
   static const uint8_t mip_colours[TEXTURE_LAYERS][TEXTURE_LEVELS][4] = {
      {
         {0x00, 0x00, 0x00, 0x00},
         {0x00, 0x00, 0x00, 0x00},
         {0x00, 0x00, 0x00, 0x00},
      },
   };
   static const uint8_t checker_colour[4] = {0x00, 0xc0, 0x00, 0xff};
#elif G17P_COMPOSITE_SRGB_SOURCE_BGRA
   /* Memory byte order is BGRA. The sampled logical mip-1 colour is
    * sRGB(64,128,192), exercising nonlinear decode and every swizzle lane. */
   static const uint8_t mip_colours[TEXTURE_LAYERS][TEXTURE_LEVELS][4] = {
      {
         {0x00, 0xff, 0xff, 0xff},
         {0x00, 0xff, 0xff, 0xff},
         {0x00, 0xff, 0xff, 0xff},
      },
      {
         {0x00, 0x00, 0xff, 0xff},
         {0x00, 0x00, 0x00, 0x00},
         {0xff, 0x00, 0x00, 0xff},
      },
   };
   static const uint8_t checker_colour[4] = {0xc0, 0x80, 0x40, 0xff};
#elif G17P_COMPOSITE_SRGB_SOURCE
   static const uint8_t mip_colours[TEXTURE_LAYERS][TEXTURE_LEVELS][4] = {
      {
         {0xff, 0xff, 0x00, 0xff},
         {0xff, 0xff, 0x00, 0xff},
         {0xff, 0xff, 0x00, 0xff},
      },
      {
         {0xff, 0x00, 0x00, 0xff},
         {0x00, 0x00, 0x00, 0x00},
         {0x00, 0x00, 0xff, 0xff},
      },
   };
   static const uint8_t checker_colour[4] = {0x40, 0x80, 0xc0, 0xff};
#else
   static const uint8_t mip_colours[TEXTURE_LAYERS][TEXTURE_LEVELS][4] = {
      {
         {0xff, 0xff, 0x00, 0xff},
         {0xff, 0xff, 0x00, 0xff},
         {0xff, 0xff, 0x00, 0xff},
      },
      {
         {0xff, 0x00, 0x00, 0xff},
         {0x00, 0x00, 0x00, 0x00},
         {0x00, 0x00, 0xff, 0xff},
      },
   };
   static const uint8_t checker_colour[4] = {0, 192, 0, 255};
#endif
   const size_t mip_offsets[TEXTURE_LEVELS] = {
      TEXTURE_GUARD_BYTES,
      TEXTURE_GUARD_BYTES + TEXTURE_LEVEL0_BYTES,
      TEXTURE_GUARD_BYTES + TEXTURE_LEVEL0_BYTES + TEXTURE_LEVEL1_BYTES,
   };
   const size_t mip_pixels[TEXTURE_LEVELS] = {
      TEXTURE_WIDTH * TEXTURE_HEIGHT,
      (TEXTURE_WIDTH / 2) * (TEXTURE_HEIGHT / 2),
      (TEXTURE_WIDTH / 4) * (TEXTURE_HEIGHT / 4),
   };
   for (unsigned level = 0; level < TEXTURE_LEVELS; ++level) {
      for (unsigned layer = 0; layer < TEXTURE_LAYERS; ++layer) {
         uint8_t *dst = texture_upload_expected + mip_offsets[level] +
                        layer * mip_pixels[level] * TEXTURE_PIXEL_BYTES;
         for (size_t pixel = 0; pixel < mip_pixels[level]; ++pixel) {
            const uint8_t *colour = mip_colours[layer][level];
            if ((G17P_COMPOSITE_GENERATE_MIPS ? layer == 0 : layer == 1) &&
                level == (G17P_COMPOSITE_GENERATE_MIPS ? 0 : 1)) {
               const unsigned level_width = TEXTURE_WIDTH >> level;
               const unsigned x = pixel % level_width;
               const unsigned y = pixel / level_width;
               const unsigned checker_x = G17P_COMPOSITE_GENERATE_MIPS ? x / 2 : x;
               const unsigned checker_y = G17P_COMPOSITE_GENERATE_MIPS ? y / 2 : y;
               if ((checker_x ^ checker_y) & 1)
                  colour = checker_colour;
            }
            memcpy(dst + pixel * TEXTURE_PIXEL_BYTES, colour,
                   TEXTURE_PIXEL_BYTES);
         }
      }
   }
#elif G17P_TWO_SAMPLERS
   /* Layer 0 is tex_a and layer 1 is tex_b. Every level of both layers is
    * uniform, so whichever mip the implicit derivatives select the sampled
    * colours — and therefore the product — are exact. */
   static const uint8_t source_colours[TEXTURE_LAYERS][4] = {
      G17P_TWO_SAMPLERS_SOURCE_A,
      G17P_TWO_SAMPLERS_SOURCE_B,
   };
   const size_t mip_offsets[TEXTURE_LEVELS] = {
      TEXTURE_GUARD_BYTES,
      TEXTURE_GUARD_BYTES + TEXTURE_LEVEL0_BYTES,
      TEXTURE_GUARD_BYTES + TEXTURE_LEVEL0_BYTES + TEXTURE_LEVEL1_BYTES,
   };
   const size_t mip_pixels[TEXTURE_LEVELS] = {
      TEXTURE_WIDTH * TEXTURE_HEIGHT,
      (TEXTURE_WIDTH / 2) * (TEXTURE_HEIGHT / 2),
      (TEXTURE_WIDTH / 4) * (TEXTURE_HEIGHT / 4),
   };
   for (unsigned level = 0; level < TEXTURE_LEVELS; ++level) {
      for (unsigned layer = 0; layer < TEXTURE_LAYERS; ++layer) {
         uint8_t *dst = texture_upload_expected + mip_offsets[level] +
                        layer * mip_pixels[level] * TEXTURE_PIXEL_BYTES;
         for (size_t pixel = 0; pixel < mip_pixels[level]; ++pixel)
            memcpy(dst + pixel * TEXTURE_PIXEL_BYTES, source_colours[layer],
                   TEXTURE_PIXEL_BYTES);
      }
   }
#else
   static const uint8_t mip_colours[TEXTURE_LEVELS][4] = {
#if G17P_SATURATE_TEXTURE
      {0xff, 0xff, 0xff, 0xff},
      {0xff, 0xff, 0xff, 0xff},
      {0xff, 0xff, 0xff, 0xff},
#else
      {0xff, 0x00, 0x00, 0xff},
      {0x00, 0xff, 0x00, 0xff},
      {0x00, 0x00, 0xff, 0xff},
#endif
   };
   const size_t mip_offsets[TEXTURE_LEVELS] = {
      TEXTURE_GUARD_BYTES,
      TEXTURE_GUARD_BYTES + TEXTURE_LEVEL0_BYTES,
      TEXTURE_GUARD_BYTES + TEXTURE_LEVEL0_BYTES + TEXTURE_LEVEL1_BYTES,
   };
   const size_t mip_pixels[TEXTURE_LEVELS] = {64 * 64, 32 * 32, 16 * 16};
   for (unsigned level = 0; level < TEXTURE_LEVELS; ++level) {
      uint8_t *dst = texture_upload_expected + mip_offsets[level];
      for (size_t pixel = 0; pixel < mip_pixels[level]; ++pixel) {
         memcpy(dst + pixel * TEXTURE_PIXEL_BYTES, mip_colours[level],
                TEXTURE_PIXEL_BYTES);
      }
   }
#endif
#if G17P_TEXTURE_GATHER
   struct gather_texel {
      unsigned x, y;
      uint8_t red;
   };
   static const struct gather_texel gather_texels[] = {
      {31, 31, 0},
      {32, 31, 85},
      {31, 32, 170},
      {32, 32, 255},
   };
   for (unsigned i = 0; i < sizeof(gather_texels) / sizeof(gather_texels[0]);
        ++i) {
      const struct gather_texel *texel = &gather_texels[i];
      texture_upload_expected[TEXTURE_GUARD_BYTES +
                              (texel->y * TEXTURE_WIDTH + texel->x) * 4] =
         texel->red;
   }
#endif
   struct mapped_buffer texture_upload = create_mapped_buffer(
      device, &memory_properties, sizeof(texture_upload_expected),
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "texture-oracle upload");
   memcpy(texture_upload.map, texture_upload_expected,
          sizeof(texture_upload_expected));
#if G17P_COMPOSITE_GENERATE_MIPS
   struct mapped_buffer generated_mip_readback = create_mapped_buffer(
      device, &memory_properties, GENERATED_READBACK_BYTES,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT, "generated mip1 readback");
   memset(generated_mip_readback.map, 0xa5, GENERATED_READBACK_BYTES);
#endif
#endif

   const VkImageCreateInfo texture_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
#if G17P_CUBE_SAMPLE
      .flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
#endif
      .imageType = G17P_TEXTURE_3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
      .format = TEXTURE_FORMAT,
      .extent = {TEXTURE_WIDTH, TEXTURE_HEIGHT, TEXTURE_DEPTH},
      .mipLevels = TEXTURE_LEVELS,
      .arrayLayers = TEXTURE_LAYERS,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT |
               (G17P_COMPOSITE_GENERATE_MIPS ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0) |
               (G17P_TEXTURE_3D ? VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT
                                : VK_IMAGE_USAGE_TRANSFER_DST_BIT),
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage texture;
   check(vkCreateImage(device, &texture_info, NULL, &texture),
         "vkCreateImage(texture)");
   VkMemoryRequirements texture_requirements;
   vkGetImageMemoryRequirements(device, texture, &texture_requirements);
   const VkMemoryAllocateInfo texture_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = texture_requirements.size,
      .memoryTypeIndex = find_memory_type(
         &memory_properties, texture_requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory texture_memory;
   check(vkAllocateMemory(device, &texture_memory_info, NULL, &texture_memory),
         "vkAllocateMemory(texture)");
   check(vkBindImageMemory(device, texture, texture_memory, 0),
         "vkBindImageMemory(texture)");
#if G17P_TEXTURE_3D
   const VkHostImageLayoutTransitionInfoEXT host_to_general = {
      .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT,
      .image = texture,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .subresourceRange = {
         .aspectMask = TEXTURE_ASPECT,
         .levelCount = 1,
         .layerCount = 1,
      },
   };
   check(vkTransitionImageLayoutEXT(device, 1, &host_to_general),
         "vkTransitionImageLayoutEXT(texture3d upload)");
   const VkMemoryToImageCopyEXT host_texture_regions[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,
         .pHostPointer = texture_upload_expected + TEXTURE_GUARD_BYTES,
         .memoryRowLength = TEXTURE_WIDTH,
         .memoryImageHeight = TEXTURE_HEIGHT,
         .imageSubresource = {TEXTURE_ASPECT, 0, 0, 1},
         .imageExtent = {TEXTURE_WIDTH, TEXTURE_HEIGHT, TEXTURE_DEPTH},
      }, {
         .sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,
         .pHostPointer = texture_upload_expected + TEXTURE_GUARD_BYTES +
                         TEXTURE_LEVEL0_BYTES,
         .memoryRowLength = 32,
         .memoryImageHeight = 32,
         .imageSubresource = {TEXTURE_ASPECT, 1, 0, 1},
         .imageExtent = {32, 32, 2},
      },
   };
   const VkCopyMemoryToImageInfoEXT host_texture_copy = {
      .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,
      .dstImage = texture,
      .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .regionCount = 2,
      .pRegions = host_texture_regions,
   };
   check(vkCopyMemoryToImageEXT(device, &host_texture_copy),
         "vkCopyMemoryToImageEXT(texture3d)");
   VkHostImageLayoutTransitionInfoEXT host_to_shader = host_to_general;
   host_to_shader.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
   host_to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   check(vkTransitionImageLayoutEXT(device, 1, &host_to_shader),
         "vkTransitionImageLayoutEXT(texture3d sample)");
#endif
#endif

   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = IMAGE_FORMAT,
      .extent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = IMAGE_LAYERS,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | READBACK_IMAGE_USAGE,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   check(vkCreateImage(device, &image_info, NULL, &image), "vkCreateImage");
   VkMemoryRequirements image_requirements;
   vkGetImageMemoryRequirements(device, image, &image_requirements);
   const uint32_t image_memory_type =
      find_memory_type(&memory_properties, image_requirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   const VkMemoryAllocateInfo image_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = image_requirements.size,
      .memoryTypeIndex = image_memory_type,
   };
   VkDeviceMemory image_memory;
   check(vkAllocateMemory(device, &image_memory_info, NULL, &image_memory),
         "vkAllocateMemory(image)");
   check(vkBindImageMemory(device, image, image_memory, 0),
         "vkBindImageMemory");

#if G17P_RGBA8_MRT
   VkImage image1;
   check(vkCreateImage(device, &image_info, NULL, &image1),
         "vkCreateImage(attachment1)");
   VkMemoryRequirements image1_requirements;
   vkGetImageMemoryRequirements(device, image1, &image1_requirements);
   const uint32_t image1_memory_type =
      find_memory_type(&memory_properties, image1_requirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   const VkMemoryAllocateInfo image1_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = image1_requirements.size,
      .memoryTypeIndex = image1_memory_type,
   };
   VkDeviceMemory image1_memory;
   check(vkAllocateMemory(device, &image1_memory_info, NULL, &image1_memory),
         "vkAllocateMemory(attachment1)");
   check(vkBindImageMemory(device, image1, image1_memory, 0),
         "vkBindImageMemory(attachment1)");
#endif

#if G17P_MSAA_SAMPLES > 1
   VkImageCreateInfo msaa_image_info = image_info;
   msaa_image_info.samples = (VkSampleCountFlagBits)G17P_MSAA_SAMPLES;
   msaa_image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   VkImage msaa_image;
   check(vkCreateImage(device, &msaa_image_info, NULL, &msaa_image),
         "vkCreateImage(msaa2)");
   VkMemoryRequirements msaa_image_requirements;
   vkGetImageMemoryRequirements(device, msaa_image, &msaa_image_requirements);
   const uint32_t msaa_image_memory_type =
      find_memory_type(&memory_properties,
                       msaa_image_requirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   const VkMemoryAllocateInfo msaa_image_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = msaa_image_requirements.size,
      .memoryTypeIndex = msaa_image_memory_type,
   };
   VkDeviceMemory msaa_image_memory;
   check(vkAllocateMemory(device, &msaa_image_memory_info, NULL,
                          &msaa_image_memory),
         "vkAllocateMemory(msaa2)");
   check(vkBindImageMemory(device, msaa_image, msaa_image_memory, 0),
         "vkBindImageMemory(msaa2)");
#endif

#if G17P_HOST_IMAGE_COPY
   uint8_t readback_storage[READBACK_BYTES];
   uint8_t *readback_map = readback_storage;
#else
   const VkBufferCreateInfo readback_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = READBACK_BYTES,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer readback;
   check(vkCreateBuffer(device, &readback_info, NULL, &readback),
         "vkCreateBuffer(readback)");
   VkMemoryRequirements readback_requirements;
   vkGetBufferMemoryRequirements(device, readback, &readback_requirements);
   const uint32_t readback_memory_type = find_memory_type(
      &memory_properties, readback_requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   const VkMemoryAllocateInfo readback_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = readback_requirements.size,
      .memoryTypeIndex = readback_memory_type,
   };
   VkDeviceMemory readback_memory;
   check(vkAllocateMemory(device, &readback_memory_info, NULL,
                          &readback_memory),
         "vkAllocateMemory(readback)");
   check(vkBindBufferMemory(device, readback, readback_memory, 0),
         "vkBindBufferMemory(readback)");
   uint8_t *readback_map = NULL;
   check(vkMapMemory(device, readback_memory, 0, READBACK_BYTES, 0,
                     (void **)&readback_map),
         "vkMapMemory(readback)");
#endif
   memset(readback_map, 0xa5, READBACK_BYTES);

   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = (G17P_MULTIVIEW || G17P_LAYER_OUTPUT)
                     ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                     : VK_IMAGE_VIEW_TYPE_2D,
      .format = IMAGE_FORMAT,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1,
         .layerCount = IMAGE_LAYERS,
      },
   };
   VkImageView view;
   check(vkCreateImageView(device, &view_info, NULL, &view),
         "vkCreateImageView");

#if G17P_RGBA8_MRT
   VkImageViewCreateInfo view1_info = view_info;
   view1_info.image = image1;
   VkImageView view1;
   check(vkCreateImageView(device, &view1_info, NULL, &view1),
         "vkCreateImageView(attachment1)");
#endif

#if G17P_TEXTURE_ORACLE
#if G17P_YCBCR
   const VkSamplerYcbcrConversionCreateInfo ycbcr_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
      .format = TEXTURE_FORMAT,
      .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
      .ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
      .components = {
         VK_COMPONENT_SWIZZLE_IDENTITY,
         VK_COMPONENT_SWIZZLE_IDENTITY,
         VK_COMPONENT_SWIZZLE_IDENTITY,
         VK_COMPONENT_SWIZZLE_IDENTITY,
      },
      .xChromaOffset = VK_CHROMA_LOCATION_MIDPOINT,
      .yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT,
      .chromaFilter = VK_FILTER_NEAREST,
      .forceExplicitReconstruction = VK_FALSE,
   };
   VkSamplerYcbcrConversion ycbcr_conversion;
   check(vkCreateSamplerYcbcrConversion(device, &ycbcr_info, NULL,
                                         &ycbcr_conversion),
         "vkCreateSamplerYcbcrConversion");
   const VkSamplerYcbcrConversionInfo sampler_ycbcr_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
      .conversion = ycbcr_conversion,
   };
#endif
#if G17P_TEXTURE_MIN_LOD
   const VkImageViewMinLodCreateInfoEXT view_clamp = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_MIN_LOD_CREATE_INFO_EXT,
      .minLod = view_min_lod,
   };
#endif
   const VkImageViewCreateInfo texture_view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
#if G17P_TEXTURE_MIN_LOD
      .pNext = &view_clamp,
#elif G17P_YCBCR
      .pNext = &sampler_ycbcr_info,
#endif
      .image = texture,
      .viewType = G17P_COMPOSITE_DESCRIPTOR_ARRAY
                     ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                  : G17P_CUBE_SAMPLE ? VK_IMAGE_VIEW_TYPE_CUBE
                  : G17P_TEXTURE_3D ? VK_IMAGE_VIEW_TYPE_3D
                                    : VK_IMAGE_VIEW_TYPE_2D,
      .format = TEXTURE_FORMAT,
      .subresourceRange = {
         .aspectMask = TEXTURE_ASPECT,
         .baseMipLevel = G17P_COMPOSITE_ARRAY_NEAREST_EXACT ? 1 : 0,
         .levelCount = G17P_COMPOSITE_ARRAY_NEAREST_EXACT ? 1 : TEXTURE_LEVELS,
         .layerCount = TEXTURE_LAYERS,
      },
   };
#if G17P_TEXTURE_ALPHA_COMPOSITE || G17P_TWO_SAMPLERS
   VkImageView texture_views[2];
   for (uint32_t source = 0; source < 2; ++source) {
      VkImageViewCreateInfo source_view_info = texture_view_info;
      source_view_info.subresourceRange.baseArrayLayer =
         G17P_COMPOSITE_GENERATE_MIPS ? 0 : source;
      source_view_info.subresourceRange.layerCount = 1;
      check(vkCreateImageView(device, &source_view_info, NULL,
                              &texture_views[source]),
            "vkCreateImageView(composite source)");
   }
#else
   VkImageView texture_view;
   check(vkCreateImageView(device, &texture_view_info, NULL, &texture_view),
         "vkCreateImageView(texture)");
#endif
   const VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
#if G17P_YCBCR
      .pNext = &sampler_ycbcr_info,
#endif
      .magFilter = G17P_COMPOSITE_ARRAY_NEAREST_EXACT ? VK_FILTER_NEAREST
                   : G17P_TEXTURE_ALPHA_COMPOSITE ? VK_FILTER_LINEAR
                                                : VK_FILTER_NEAREST,
      .minFilter = G17P_COMPOSITE_ARRAY_NEAREST_EXACT ? VK_FILTER_NEAREST
                   : G17P_TEXTURE_ALPHA_COMPOSITE ? VK_FILTER_LINEAR
                                                : VK_FILTER_NEAREST,
      .mipmapMode = G17P_COMPOSITE_GENERATE_MIPS
                       ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                       : VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
#if G17P_SHADOW_COMPARE
      .compareEnable = VK_TRUE,
      .compareOp = VK_COMPARE_OP_LESS,
#endif
      .minLod = 0.0f,
      .maxLod = (float)(TEXTURE_LEVELS - 1),
   };
   VkSampler sampler;
   check(vkCreateSampler(device, &sampler_info, NULL, &sampler),
         "vkCreateSampler");
#if G17P_TEXTURE_ALPHA_COMPOSITE
#if G17P_COMPOSITE_DESCRIPTOR_CELL == 2
   const VkDescriptorSetLayoutBinding descriptor_bindings[] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
         .binding = 2,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
   };
#elif G17P_COMPOSITE_DESCRIPTOR_CELL == 1
   const VkDescriptorSetLayoutBinding descriptor_bindings[] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
   };
#else
   const VkDescriptorSetLayoutBinding descriptor_bindings[] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 2,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
   };
#endif
#elif G17P_TWO_SAMPLERS
   /* Two separate combined-image-sampler cells, either side by side in set 0
    * or — for the split variant — one here and an identical one in set 1. */
   const VkDescriptorSetLayoutBinding descriptor_bindings[] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
#if !G17P_TWO_SAMPLERS_SPLIT_SETS
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
#endif
   };
#else
   const VkDescriptorSetLayoutBinding descriptor_bindings[] = {{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
#if G17P_YCBCR
      .pImmutableSamplers = &sampler,
#endif
   }};
#endif
   const VkDescriptorSetLayoutCreateInfo set_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = sizeof(descriptor_bindings) /
                      sizeof(descriptor_bindings[0]),
      .pBindings = descriptor_bindings,
   };
   VkDescriptorSetLayout set_layout;
   check(vkCreateDescriptorSetLayout(device, &set_layout_info, NULL,
                                     &set_layout),
         "vkCreateDescriptorSetLayout");
#if G17P_TWO_SAMPLERS_SPLIT_SETS
   /* Set 1 repeats set 0's single binding, so the two sets are identical and
    * the only variable against the one-set control is the set index. */
   VkDescriptorSetLayout set_layout_b;
   check(vkCreateDescriptorSetLayout(device, &set_layout_info, NULL,
                                     &set_layout_b),
         "vkCreateDescriptorSetLayout(set 1)");
   const VkDescriptorSetLayout set_layouts[2] = {set_layout, set_layout_b};
#endif
#if G17P_TEXTURE_ALPHA_COMPOSITE
#if G17P_COMPOSITE_DESCRIPTOR_CELL == 2
   const VkDescriptorPoolSize pool_sizes[] = {
      {
         .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .descriptorCount = 1,
      },
      {
         .type = VK_DESCRIPTOR_TYPE_SAMPLER,
         .descriptorCount = 1,
      },
      {
         .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
         .descriptorCount = 1,
      },
   };
#else
   const VkDescriptorPoolSize pool_sizes[] = {
      {
         .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = G17P_COMPOSITE_DESCRIPTOR_CELL == 1 ? 1 : 2,
      },
      {
         .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
         .descriptorCount = 1,
      },
   };
#endif
#elif G17P_TWO_SAMPLERS
   const VkDescriptorPoolSize pool_sizes[] = {{
      .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 2,
   }};
#else
   const VkDescriptorPoolSize pool_sizes[] = {{
      .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
   }};
#endif
   const VkDescriptorPoolCreateInfo descriptor_pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = G17P_TWO_SAMPLERS_SPLIT_SETS ? 2 : 1,
      .poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
      .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool descriptor_pool;
   check(vkCreateDescriptorPool(device, &descriptor_pool_info, NULL,
                                &descriptor_pool),
         "vkCreateDescriptorPool");
#if G17P_TWO_SAMPLERS_SPLIT_SETS
   const VkDescriptorSetAllocateInfo set_allocate_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 2,
      .pSetLayouts = set_layouts,
   };
   VkDescriptorSet descriptor_sets[2];
   check(vkAllocateDescriptorSets(device, &set_allocate_info,
                                  descriptor_sets),
         "vkAllocateDescriptorSets");
#else
   const VkDescriptorSetAllocateInfo set_allocate_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkDescriptorSet descriptor_set;
   check(vkAllocateDescriptorSets(device, &set_allocate_info,
                                  &descriptor_set),
         "vkAllocateDescriptorSets");
#endif
#if G17P_TEXTURE_ALPHA_COMPOSITE
   const VkDescriptorImageInfo descriptor_images[2] = {
      {
         .sampler = sampler,
         .imageView = texture_views[0],
         .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      },
      {
         .sampler = sampler,
         .imageView = texture_views[1],
         .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      },
   };
   const VkDescriptorBufferInfo descriptor_uniform = {
      .buffer = composite_uniform.buffer,
      .offset = COMPOSITE_UNIFORM_OFFSET,
      .range = sizeof(struct composite_params),
   };
#if G17P_COMPOSITE_DESCRIPTOR_CELL == 2
   const VkWriteDescriptorSet descriptor_writes[] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
         .pImageInfo = &descriptor_images[1],
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .pImageInfo = &descriptor_images[1],
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
         .pBufferInfo = &descriptor_uniform,
      },
   };
#elif G17P_COMPOSITE_DESCRIPTOR_CELL == 1
   const VkWriteDescriptorSet descriptor_writes[] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &descriptor_images[1],
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
         .pBufferInfo = &descriptor_uniform,
      },
   };
#else
   const VkWriteDescriptorSet descriptor_writes[] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 0,
         .descriptorCount = 2,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = descriptor_images,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
         .pBufferInfo = &descriptor_uniform,
      },
   };
#endif
#elif G17P_TWO_SAMPLERS
   const VkDescriptorImageInfo descriptor_images[2] = {
      {
         .sampler = sampler,
         .imageView = texture_views[0],
         .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      },
      {
         .sampler = sampler,
         .imageView = texture_views[1],
         .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      },
   };
   const VkWriteDescriptorSet descriptor_writes[] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
#if G17P_TWO_SAMPLERS_SPLIT_SETS
         .dstSet = descriptor_sets[0],
#else
         .dstSet = descriptor_set,
#endif
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &descriptor_images[0],
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
#if G17P_TWO_SAMPLERS_SPLIT_SETS
         .dstSet = descriptor_sets[1],
         .dstBinding = 0,
#else
         .dstSet = descriptor_set,
         .dstBinding = 1,
#endif
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &descriptor_images[1],
      },
   };
#else
   const VkDescriptorImageInfo descriptor_image = {
      .sampler = sampler,
      .imageView = texture_view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   const VkWriteDescriptorSet descriptor_writes[] = {{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &descriptor_image,
   }};
#endif
   vkUpdateDescriptorSets(device,
                          sizeof(descriptor_writes) /
                             sizeof(descriptor_writes[0]),
                          descriptor_writes, 0, NULL);
#endif

#if G17P_MSAA_SAMPLES > 1
   VkImageViewCreateInfo msaa_view_info = view_info;
   msaa_view_info.image = msaa_image;
   VkImageView msaa_view;
   check(vkCreateImageView(device, &msaa_view_info, NULL, &msaa_view),
         "vkCreateImageView(msaa2)");
#endif

#if G17P_RGBA8_MRT
   const VkAttachmentDescription attachments[] = {
      {
         .format = IMAGE_FORMAT,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      },
      {
         .format = IMAGE_FORMAT,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      },
   };
#else
   const VkAttachmentDescription attachments[] = {{
      .format = IMAGE_FORMAT,
      .samples = (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
#if G17P_MSAA_SAMPLES > 1
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
#else
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
#endif
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
#if G17P_MSAA_SAMPLES > 1
      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
#endif
   },
#if G17P_MSAA_SAMPLES > 1
   {
      .format = IMAGE_FORMAT,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
   },
#endif
   };
#endif
#if G17P_RGBA8_MRT
   const VkAttachmentReference color_refs[] = {
      {.attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
      {.attachment = 1, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
   };
#else
   const VkAttachmentReference color_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
#endif
#if G17P_MSAA_SAMPLES > 1
   const VkAttachmentReference resolve_ref = {
      .attachment = 1,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
#endif
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
#if G17P_RGBA8_MRT
      .colorAttachmentCount = 2,
      .pColorAttachments = color_refs,
#else
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
#endif
#if G17P_MSAA_SAMPLES > 1
      .pResolveAttachments = &resolve_ref,
#endif
   };
   const VkSubpassDependency dependencies[] = {
      {
         .srcSubpass = VK_SUBPASS_EXTERNAL,
         .dstSubpass = 0,
         .srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      },
      {
         .srcSubpass = 0,
         .dstSubpass = VK_SUBPASS_EXTERNAL,
         .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      },
   };
#if G17P_MULTIVIEW
   const uint32_t multiview_mask = 0x3;
   const VkRenderPassMultiviewCreateInfo multiview_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
      .subpassCount = 1,
      .pViewMasks = &multiview_mask,
      .correlationMaskCount = 1,
      .pCorrelationMasks = &multiview_mask,
   };
#endif
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
#if G17P_MULTIVIEW
      .pNext = &multiview_info,
#endif
      .attachmentCount = sizeof(attachments) / sizeof(attachments[0]),
      .pAttachments = attachments,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 2,
      .pDependencies = dependencies,
   };
   VkRenderPass render_pass;
   check(vkCreateRenderPass(device, &render_pass_info, NULL, &render_pass),
         "vkCreateRenderPass");

#if G17P_RGBA8_MRT
   VkAttachmentDescription load_attachment = attachments[0];
   load_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
   load_attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   load_attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   const VkAttachmentReference load_color_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkSubpassDescription load_subpass = subpass;
   load_subpass.colorAttachmentCount = 1;
   load_subpass.pColorAttachments = &load_color_ref;
   VkRenderPassCreateInfo load_render_pass_info = render_pass_info;
   load_render_pass_info.attachmentCount = 1;
   load_render_pass_info.pAttachments = &load_attachment;
   load_render_pass_info.pSubpasses = &load_subpass;
   VkRenderPass load_render_pass;
   check(vkCreateRenderPass(device, &load_render_pass_info, NULL,
                            &load_render_pass),
         "vkCreateRenderPass(load-additive)");
#endif

   const VkShaderModuleCreateInfo vert_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(G17P_VERT_SPV),
      .pCode = G17P_VERT_SPV,
   };
   const VkShaderModuleCreateInfo frag_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(G17P_FRAG_SPV),
      .pCode = G17P_FRAG_SPV,
   };
   VkShaderModule vert_module, frag_module;
   check(vkCreateShaderModule(device, &vert_module_info, NULL, &vert_module),
         "vkCreateShaderModule(vertex)");
   check(vkCreateShaderModule(device, &frag_module_info, NULL, &frag_module),
         "vkCreateShaderModule(fragment)");

#if G17P_SECONDARY_COMMAND_BUFFER_MULTI
   const VkPushConstantRange quadrant_color_range = {
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      .size = sizeof(secondary_quadrant_colors[0]),
   };
#endif
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
#if G17P_TWO_SAMPLERS_SPLIT_SETS
      .setLayoutCount = 2,
      .pSetLayouts = set_layouts,
#elif G17P_TEXTURE_ORACLE
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
#endif
#if G17P_SECONDARY_COMMAND_BUFFER_MULTI
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &quadrant_color_range,
#endif
   };
   VkPipelineLayout pipeline_layout;
   check(vkCreatePipelineLayout(device, &layout_info, NULL, &pipeline_layout),
         "vkCreatePipelineLayout");

   const VkPipelineShaderStageCreateInfo stages[] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vert_module,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = frag_module,
         .pName = "main",
      },
   };
#if G17P_VERTEX_FORMAT
   const VkVertexInputBindingDescription vertex_binding = {
      .binding = VERTEX_FORMAT_BINDING,
      .stride = VERTEX_FORMAT_STRIDE,
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription vertex_attribute = {
      .location = 0,
      .binding = VERTEX_FORMAT_BINDING,
      .format = VERTEX_INPUT_FORMAT,
      .offset = VERTEX_FORMAT_ATTRIBUTE_OFFSET,
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &vertex_attribute,
   };
#elif G17P_INSTANCE_RATE
   const VkVertexInputBindingDescription vertex_bindings[] = {
      {
         .binding = 0,
         .stride = INSTANCE_VERTEX_STRIDE,
         .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
      },
      {
         .binding = 1,
         .stride = INSTANCE_DATA_STRIDE,
         .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE,
      },
   };
   const VkVertexInputAttributeDescription vertex_attributes[] = {
      {
         .location = 0,
         .binding = 0,
         .format = VK_FORMAT_R32G32_SFLOAT,
         .offset = INSTANCE_VERTEX_ATTRIBUTE_OFFSET,
      },
      {
         .location = 1,
         .binding = 1,
         .format = VK_FORMAT_R32G32B32A32_SFLOAT,
         .offset = INSTANCE_DATA_TRANSFORM_OFFSET,
      },
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount =
         sizeof(vertex_bindings) / sizeof(vertex_bindings[0]),
      .pVertexBindingDescriptions = vertex_bindings,
      .vertexAttributeDescriptionCount =
         sizeof(vertex_attributes) / sizeof(vertex_attributes[0]),
      .pVertexAttributeDescriptions = vertex_attributes,
   };
#elif G17P_TEXTURE_ALPHA_COMPOSITE
   const VkVertexInputBindingDescription vertex_binding = {
      .binding = 0,
      .stride = sizeof(struct composite_vertex),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription vertex_attributes[] = {
      {
         .location = 0,
         .binding = 0,
         .format = VK_FORMAT_R32G32_SFLOAT,
         .offset = 0,
      },
      {
         .location = 1,
         .binding = 0,
         .format = VK_FORMAT_R32_UINT,
         .offset = 2 * sizeof(float),
      },
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount =
         sizeof(vertex_attributes) / sizeof(vertex_attributes[0]),
      .pVertexAttributeDescriptions = vertex_attributes,
   };
#else
   const VkVertexInputBindingDescription vertex_binding = {
      .binding = 0,
      .stride = 2 * sizeof(float),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription vertex_attribute = {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &vertex_attribute,
   };
#endif
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = G17P_BRESENHAM_LINES
                     ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
                     : G17P_INDEX_RESTART
                          ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP
                          : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = G17P_INDEX_RESTART,
   };
#if G17P_MULTI_VIEWPORT
   const VkViewport viewports[4] = {
      {
         .width = IMAGE_WIDTH / 2,
         .height = IMAGE_HEIGHT / 2,
         .minDepth = 0.0f,
         .maxDepth = 1.0f,
      },
      {
         .x = IMAGE_WIDTH / 2,
         .width = IMAGE_WIDTH / 2,
         .height = IMAGE_HEIGHT / 2,
         .minDepth = 0.0f,
         .maxDepth = 1.0f,
      },
      {
         .y = IMAGE_HEIGHT / 2,
         .width = IMAGE_WIDTH / 2,
         .height = IMAGE_HEIGHT / 2,
         .minDepth = 0.0f,
         .maxDepth = 1.0f,
      },
      {
         .x = IMAGE_WIDTH / 2,
         .y = IMAGE_HEIGHT / 2,
         .width = IMAGE_WIDTH / 2,
         .height = IMAGE_HEIGHT / 2,
         .minDepth = 0.0f,
         .maxDepth = 1.0f,
      },
   };
   const VkRect2D scissors[4] = {
      {.extent = {IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2}},
      {
         .offset = {IMAGE_WIDTH / 2, 0},
         .extent = {IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2},
      },
      {
         .offset = {0, IMAGE_HEIGHT / 2},
         .extent = {IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2},
      },
      {
         .offset = {IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2},
         .extent = {IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2},
      },
   };
#else
   const VkViewport viewport = {
      .width = IMAGE_WIDTH,
      .height = IMAGE_HEIGHT,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
   };
   const VkRect2D scissor = {
      .extent = {
         G17P_RGBA8_MRT ? IMAGE_WIDTH / 2 : IMAGE_WIDTH,
         IMAGE_HEIGHT,
      },
   };
#endif
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
#if G17P_MULTI_VIEWPORT
      .viewportCount = 4,
#if G17P_MULTI_VIEWPORT_DYNAMIC_VIEWPORT
      .pViewports = NULL,
#else
      .pViewports = viewports,
#endif
      .scissorCount = 4,
#if G17P_MULTI_VIEWPORT_DYNAMIC_SCISSOR
      .pScissors = NULL,
#else
      .pScissors = scissors,
#endif
#else
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
#endif
   };
#if G17P_BRESENHAM_LINES
   const VkPipelineRasterizationLineStateCreateInfoKHR line_raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_KHR,
      .lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR,
   };
#endif
   const VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
#if G17P_BRESENHAM_LINES
      .pNext = &line_raster,
#endif
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .rasterizerDiscardEnable = G17P_RASTER_DISCARD,
#if G17P_BRESENHAM_LINES
      .lineWidth = 3.0f,
#else
      .lineWidth = 1.0f,
#endif
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
      .sampleShadingEnable =
         G17P_SAMPLE_RATE_SHADING || G17P_MSAA_COVERAGE == 1,
      .minSampleShading =
         (G17P_SAMPLE_RATE_SHADING || G17P_MSAA_COVERAGE == 1) ? 1.0f : 0.0f,
   };
#if G17P_ADVANCED_BLEND
   const VkPipelineColorBlendAdvancedStateCreateInfoEXT advanced_blend = {
      .sType =
         VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_ADVANCED_STATE_CREATE_INFO_EXT,
      .srcPremultiplied = VK_FALSE,
      .dstPremultiplied = VK_FALSE,
      .blendOverlap = VK_BLEND_OVERLAP_CONJOINT_EXT,
   };
#endif
   const VkPipelineColorBlendAttachmentState blend_attachment = {
      .blendEnable =
         G17P_INDEX_RESTART || G17P_BLEND_CONSTANT || G17P_ADVANCED_BLEND ||
         G17P_TEXTURE_ALPHA_COMPOSITE,
      .srcColorBlendFactor = G17P_BLEND_CONSTANT
                                ? VK_BLEND_FACTOR_CONSTANT_COLOR
                                : VK_BLEND_FACTOR_ONE,
      .dstColorBlendFactor =
         G17P_TEXTURE_ALPHA_COMPOSITE
            ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
            : (G17P_BLEND_CONSTANT || G17P_ADVANCED_BLEND)
                 ? VK_BLEND_FACTOR_ZERO
                 : VK_BLEND_FACTOR_ONE,
      .colorBlendOp =
         G17P_ADVANCED_BLEND ? VK_BLEND_OP_MULTIPLY_EXT : VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = G17P_BLEND_CONSTANT
                                ? VK_BLEND_FACTOR_CONSTANT_ALPHA
                                : VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor =
         G17P_TEXTURE_ALPHA_COMPOSITE
            ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
            : (G17P_BLEND_CONSTANT || G17P_ADVANCED_BLEND)
                 ? VK_BLEND_FACTOR_ZERO
                 : VK_BLEND_FACTOR_ONE,
      .alphaBlendOp =
         G17P_ADVANCED_BLEND ? VK_BLEND_OP_MULTIPLY_EXT : VK_BLEND_OP_ADD,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
#if G17P_RGBA8_MRT
   const VkPipelineColorBlendAttachmentState blend_attachments[2] = {
      blend_attachment, blend_attachment,
   };
#endif
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
#if G17P_ADVANCED_BLEND
      .pNext = &advanced_blend,
#endif
      .logicOpEnable = G17P_LOGIC_OP_XOR,
      .logicOp = VK_LOGIC_OP_XOR,
      .attachmentCount = G17P_RGBA8_MRT ? 2 : 1,
#if G17P_RGBA8_MRT
      .pAttachments = blend_attachments,
#else
      .pAttachments = &blend_attachment,
#endif
      .blendConstants = {0.25f, 0.5f, 0.75f, 0.125f},
   };
#if G17P_DYNAMIC_BLEND || G17P_TEXTURE_ALPHA_COMPOSITE ||                 \
   G17P_MULTI_VIEWPORT_DYNAMIC_VIEWPORT ||                               \
   G17P_MULTI_VIEWPORT_DYNAMIC_SCISSOR || G17P_MSAA_COVERAGE ||          \
   G17P_DYNAMIC_VERTEX_INPUT
   const VkDynamicState dynamic_states[] = {
#if G17P_MULTI_VIEWPORT_DYNAMIC_VIEWPORT
      VK_DYNAMIC_STATE_VIEWPORT,
#endif
#if G17P_MULTI_VIEWPORT_DYNAMIC_SCISSOR
      VK_DYNAMIC_STATE_SCISSOR,
#elif G17P_DYNAMIC_BLEND || G17P_TEXTURE_ALPHA_COMPOSITE ||              \
      G17P_MSAA_COVERAGE == 2 || G17P_MSAA_COVERAGE == 3
      VK_DYNAMIC_STATE_SCISSOR,
#endif
#if G17P_DYNAMIC_BLEND
      VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,
      VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
      VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT,
#endif
#if G17P_DYNAMIC_VERTEX_INPUT
      VK_DYNAMIC_STATE_VERTEX_INPUT_EXT,
#endif
#if G17P_MSAA_COVERAGE == 1 || G17P_MSAA_COVERAGE == 3
      VK_DYNAMIC_STATE_SAMPLE_MASK_EXT,
#elif G17P_MSAA_COVERAGE == 2
      VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT,
#endif
   };
   const VkPipelineDynamicStateCreateInfo dynamic_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
      .pDynamicStates = dynamic_states,
   };
#endif
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pColorBlendState = &blend,
#if G17P_DYNAMIC_BLEND || G17P_TEXTURE_ALPHA_COMPOSITE ||                 \
   G17P_MULTI_VIEWPORT_DYNAMIC_VIEWPORT ||                               \
   G17P_MULTI_VIEWPORT_DYNAMIC_SCISSOR || G17P_MSAA_COVERAGE ||          \
   G17P_DYNAMIC_VERTEX_INPUT
      .pDynamicState = &dynamic_state,
#endif
      .layout = pipeline_layout,
      .renderPass = render_pass,
   };
#if G17P_CUBE_SAMPLE
   static const float cube_directions[6][3] = {
      {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f},
   };
   static const uint32_t cube_stripe_edges[7] = {
      0, 5, 10, 15, 20, 25, IMAGE_WIDTH,
   };
   static const VkSpecializationMapEntry cube_direction_entries[3] = {
      {.constantID = 0, .offset = 0 * sizeof(float), .size = sizeof(float)},
      {.constantID = 1, .offset = 1 * sizeof(float), .size = sizeof(float)},
      {.constantID = 2, .offset = 2 * sizeof(float), .size = sizeof(float)},
   };
   VkSpecializationInfo cube_specializations[6];
   VkPipelineShaderStageCreateInfo cube_stages[6][2];
   VkViewport cube_viewports[6];
   VkRect2D cube_scissors[6];
   VkPipelineViewportStateCreateInfo cube_viewport_states[6];
   VkGraphicsPipelineCreateInfo cube_pipeline_infos[6];
   VkPipeline cube_pipelines[6];

   /* Specialization folds each direction before G17P cube lowering. Six
    * clipped draws therefore test face routing without fragment comparisons,
    * selects, or divergent control flow obscuring the result. */
   for (uint32_t face = 0; face < 6; ++face) {
      cube_specializations[face] = (VkSpecializationInfo) {
         .mapEntryCount = 3,
         .pMapEntries = cube_direction_entries,
         .dataSize = sizeof(cube_directions[face]),
         .pData = cube_directions[face],
      };
      memcpy(cube_stages[face], stages, sizeof(stages));
      cube_stages[face][1].pSpecializationInfo = &cube_specializations[face];

      const uint32_t x = cube_stripe_edges[face];
      const uint32_t width = cube_stripe_edges[face + 1] - x;
      cube_viewports[face] = (VkViewport) {
         .x = x,
         .width = width,
         .height = IMAGE_HEIGHT,
         .minDepth = 0.0f,
         .maxDepth = 1.0f,
      };
      cube_scissors[face] = (VkRect2D) {
         .offset = {(int32_t)x, 0},
         .extent = {width, IMAGE_HEIGHT},
      };
      cube_viewport_states[face] = viewport_state;
      cube_viewport_states[face].pViewports = &cube_viewports[face];
      cube_viewport_states[face].pScissors = &cube_scissors[face];

      cube_pipeline_infos[face] = pipeline_info;
      cube_pipeline_infos[face].pStages = cube_stages[face];
      cube_pipeline_infos[face].pViewportState = &cube_viewport_states[face];
   }
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 6,
                                   cube_pipeline_infos, NULL, cube_pipelines),
         "vkCreateGraphicsPipelines(cube faces)");
#else
   VkPipeline pipeline;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   NULL, &pipeline),
         "vkCreateGraphicsPipelines");
#if G17P_RGBA8_MRT
   VkPipelineColorBlendAttachmentState additive_attachments[2] = {
      blend_attachment, blend_attachment,
   };
   for (unsigned i = 0; i < 2; ++i) {
      additive_attachments[i].blendEnable = i == 0;
      additive_attachments[i].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      additive_attachments[i].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
      additive_attachments[i].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      additive_attachments[i].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      if (i == 1)
         additive_attachments[i].colorWriteMask = 0;
   }
   VkPipelineColorBlendStateCreateInfo additive_blend = blend;
   additive_blend.attachmentCount = 1;
   additive_blend.pAttachments = additive_attachments;
   VkGraphicsPipelineCreateInfo additive_pipeline_info = pipeline_info;
   additive_pipeline_info.pColorBlendState = &additive_blend;
   additive_pipeline_info.renderPass = load_render_pass;
   VkPipeline additive_pipeline;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                   &additive_pipeline_info, NULL,
                                   &additive_pipeline),
         "vkCreateGraphicsPipelines(load-additive)");
#endif
#endif

#if G17P_RGBA8_MRT
   const VkImageView framebuffer_views[] = {view, view1};
#elif G17P_MSAA_SAMPLES > 1
   const VkImageView framebuffer_views[] = {msaa_view, view};
#else
   const VkImageView framebuffer_views[] = {view};
#endif
   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount =
         sizeof(framebuffer_views) / sizeof(framebuffer_views[0]),
      .pAttachments = framebuffer_views,
      .width = IMAGE_WIDTH,
      .height = IMAGE_HEIGHT,
      .layers = G17P_LAYER_OUTPUT ? IMAGE_LAYERS : 1,
   };
   VkFramebuffer framebuffer;
   check(vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffer),
         "vkCreateFramebuffer");

#if G17P_RGBA8_MRT
   VkFramebufferCreateInfo load_framebuffer_info = framebuffer_info;
   load_framebuffer_info.renderPass = load_render_pass;
   load_framebuffer_info.attachmentCount = 1;
   load_framebuffer_info.pAttachments = &view;
   VkFramebuffer load_framebuffer;
   check(vkCreateFramebuffer(device, &load_framebuffer_info, NULL,
                             &load_framebuffer),
         "vkCreateFramebuffer(load-additive)");
#endif

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool command_pool;
   check(vkCreateCommandPool(device, &pool_info, NULL, &command_pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo command_allocate = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command_buffer;
   check(vkAllocateCommandBuffers(device, &command_allocate, &command_buffer),
         "vkAllocateCommandBuffers");
#if G17P_SECONDARY_COMMAND_BUFFER
   VkCommandBufferAllocateInfo secondary_allocate = command_allocate;
   secondary_allocate.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
#if G17P_SECONDARY_COMMAND_BUFFER_MULTI
   secondary_allocate.commandBufferCount = SECONDARY_QUADRANT_COUNT;
   VkCommandBuffer secondary_command_buffers[SECONDARY_QUADRANT_COUNT];
   check(vkAllocateCommandBuffers(device, &secondary_allocate,
                                  secondary_command_buffers),
         "vkAllocateCommandBuffers(secondary)");
#else
   VkCommandBuffer secondary_command_buffer;
   check(vkAllocateCommandBuffers(device, &secondary_allocate,
                                  &secondary_command_buffer),
         "vkAllocateCommandBuffers(secondary)");
#endif
#endif
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command_buffer, &begin_info),
         "vkBeginCommandBuffer");
#if G17P_TEXTURE_ORACLE
#if !G17P_TEXTURE_3D
   VkImageMemoryBarrier texture_barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = G17P_TEXTURE_3D ? VK_ACCESS_HOST_WRITE_BIT : 0,
      .dstAccessMask = G17P_TEXTURE_3D ? VK_ACCESS_SHADER_READ_BIT
                                      : VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = G17P_TEXTURE_3D ? VK_IMAGE_LAYOUT_PREINITIALIZED
                                   : VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = G17P_TEXTURE_3D ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                   : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = texture,
      .subresourceRange = {
         .aspectMask = TEXTURE_ASPECT,
         .levelCount = TEXTURE_LEVELS,
         .layerCount = TEXTURE_LAYERS,
      },
   };
   vkCmdPipelineBarrier(command_buffer,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, NULL, 0, NULL,
                        1, &texture_barrier);
#endif
#if G17P_SHADOW_COMPARE
   const VkClearDepthStencilValue depth_clear = {.depth = 0.5f};
   const VkImageSubresourceRange depth_range = {
      .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
      .levelCount = TEXTURE_LEVELS,
      .layerCount = 1,
   };
   vkCmdClearDepthStencilImage(command_buffer, texture,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               &depth_clear, 1, &depth_range);
#elif G17P_CUBE_SAMPLE
   const VkBufferImageCopy texture_copies[1] = {
      {
         .bufferOffset = TEXTURE_GUARD_BYTES,
         .imageSubresource = {TEXTURE_ASPECT, 0, 0, TEXTURE_LAYERS},
         .imageExtent = {TEXTURE_WIDTH, TEXTURE_HEIGHT, 1},
      },
   };
   vkCmdCopyBufferToImage(command_buffer, texture_upload.buffer, texture,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                          texture_copies);
#elif G17P_TEXTURE_3D
   /* Host-populated optimal storage avoids conflating 3D sampling with the
    * separate, not-yet-witnessed G17P 3D storage-image meta-copy path. */
#elif G17P_YCBCR
   const VkBufferImageCopy texture_copies[2] = {
      {
         .bufferOffset = TEXTURE_GUARD_BYTES,
         .imageSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1},
         .imageExtent = {TEXTURE_WIDTH, TEXTURE_HEIGHT, 1},
      },
      {
         .bufferOffset = TEXTURE_GUARD_BYTES + TEXTURE_LEVEL0_BYTES,
         .imageSubresource = {VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1},
         .imageExtent = {TEXTURE_WIDTH / 2, TEXTURE_HEIGHT / 2, 1},
      },
   };
   vkCmdCopyBufferToImage(command_buffer, texture_upload.buffer, texture,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 2,
                          texture_copies);
#elif G17P_COMPOSITE_GENERATE_MIPS
   const VkBufferImageCopy texture_copies[1] = {
      {
         .bufferOffset = TEXTURE_GUARD_BYTES,
         .imageSubresource = {TEXTURE_ASPECT, 0, 0, TEXTURE_LAYERS},
         .imageExtent = {TEXTURE_WIDTH, TEXTURE_HEIGHT, 1},
      },
   };
   vkCmdCopyBufferToImage(command_buffer, texture_upload.buffer, texture,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                          texture_copies);

   uint32_t mip_width = TEXTURE_WIDTH;
   uint32_t mip_height = TEXTURE_HEIGHT;
   for (uint32_t level = 0; level + 1 < TEXTURE_LEVELS; ++level) {
      const VkImageMemoryBarrier source_ready = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = texture,
         .subresourceRange = {TEXTURE_ASPECT, level, 1, 0, TEXTURE_LAYERS},
      };
      vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                           1, &source_ready);
      if (level == 0) {
         const VkBufferImageCopy pre_blit_level0_copy = {
            .bufferOffset = GENERATED_PRE_LEVEL0_OFFSET,
            .imageSubresource = {TEXTURE_ASPECT, 0, 0, 1},
            .imageExtent = {TEXTURE_WIDTH, TEXTURE_HEIGHT, 1},
         };
         vkCmdCopyImageToBuffer(command_buffer, texture,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                generated_mip_readback.buffer, 1,
                                &pre_blit_level0_copy);
      }
      const VkImageBlit blit = {
         .srcSubresource = {TEXTURE_ASPECT, level, 0, TEXTURE_LAYERS},
         .srcOffsets = {{0, 0, 0}, {mip_width, mip_height, 1}},
         .dstSubresource = {TEXTURE_ASPECT, level + 1, 0, TEXTURE_LAYERS},
         .dstOffsets = {{0, 0, 0},
                        {mip_width > 1 ? mip_width / 2 : 1,
                         mip_height > 1 ? mip_height / 2 : 1, 1}},
      };
      vkCmdBlitImage(command_buffer, texture,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, texture,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                     VK_FILTER_LINEAR);
      mip_width = mip_width > 1 ? mip_width / 2 : 1;
      mip_height = mip_height > 1 ? mip_height / 2 : 1;
   }

   const VkBufferImageCopy post_blit_level0_copy = {
      .bufferOffset = GENERATED_POST_LEVEL0_OFFSET,
      .imageSubresource = {TEXTURE_ASPECT, 0, 0, 1},
      .imageExtent = {TEXTURE_WIDTH, TEXTURE_HEIGHT, 1},
   };
   vkCmdCopyImageToBuffer(command_buffer, texture,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          generated_mip_readback.buffer, 1,
                          &post_blit_level0_copy);
   const VkBufferImageCopy generated_mip1_copy = {
      .bufferOffset = GENERATED_LEVEL1_OFFSET,
      .imageSubresource = {TEXTURE_ASPECT, 1, 0, 1},
      .imageExtent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
   };
   vkCmdCopyImageToBuffer(command_buffer, texture,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          generated_mip_readback.buffer, 1,
                          &generated_mip1_copy);
#else
   const VkBufferImageCopy texture_copies[TEXTURE_LEVELS] = {
      {
         .bufferOffset = TEXTURE_GUARD_BYTES,
         .imageSubresource = {TEXTURE_ASPECT, 0, 0, TEXTURE_LAYERS},
         .imageExtent = {TEXTURE_WIDTH, TEXTURE_HEIGHT, 1},
      },
      {
         .bufferOffset = TEXTURE_GUARD_BYTES + TEXTURE_LEVEL0_BYTES,
         .imageSubresource = {TEXTURE_ASPECT, 1, 0, TEXTURE_LAYERS},
         .imageExtent = {TEXTURE_WIDTH / 2, TEXTURE_HEIGHT / 2, 1},
      },
      {
         .bufferOffset = TEXTURE_GUARD_BYTES + TEXTURE_LEVEL0_BYTES +
                         TEXTURE_LEVEL1_BYTES,
         .imageSubresource = {TEXTURE_ASPECT, 2, 0, TEXTURE_LAYERS},
         .imageExtent = {TEXTURE_WIDTH / 4, TEXTURE_HEIGHT / 4, 1},
      },
   };
   vkCmdCopyBufferToImage(command_buffer, texture_upload.buffer, texture,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          TEXTURE_LEVELS, texture_copies);
#endif
#if G17P_COMPOSITE_GENERATE_MIPS
   VkImageMemoryBarrier sampled_mips[TEXTURE_LEVELS];
   for (uint32_t level = 0; level < TEXTURE_LEVELS; ++level) {
      sampled_mips[level] = (VkImageMemoryBarrier) {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = level + 1 < TEXTURE_LEVELS
                            ? VK_ACCESS_TRANSFER_READ_BIT
                            : VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
         .oldLayout = level + 1 < TEXTURE_LEVELS
                         ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                         : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = texture,
         .subresourceRange = {TEXTURE_ASPECT, level, 1, 0, TEXTURE_LAYERS},
      };
   }
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0,
                        NULL, TEXTURE_LEVELS, sampled_mips);
#elif !G17P_TEXTURE_3D
   texture_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   texture_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
   texture_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   texture_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0,
                        NULL, 1, &texture_barrier);
#endif
#endif
#if G17P_RGBA8_MRT
   const VkClearValue clears[2] = {
      {.color = {.float32 = {0.0625f, 0.125f, 0.1875f, 0.25f}}},
      {.color = {.float32 = {0.3125f, 0.375f, 0.4375f, 0.5f}}},
   };
#elif G17P_TEXTURE_ALPHA_COMPOSITE
   const VkClearValue clear = {
      .color = {.float32 = {128.0f / 255.0f, 64.0f / 255.0f,
                            128.0f / 255.0f, 1.0f}},
   };
#elif G17P_DYNAMIC_BLEND || G17P_ADVANCED_BLEND
   const VkClearValue clear = {
      .color = {.float32 = {0.25f, 0.5f, 0.75f, 1.0f}},
   };
#elif G17P_R32_SINT || G17P_RG16_SINT || G17P_RG32_SINT ||              \
   G17P_RG8_SINT || G17P_RGBA8_SINT || G17P_R8_SINT ||                \
   G17P_RGBA16_SINT || G17P_RGBA32_SINT || G17P_R16_SINT
   const VkClearValue clear = {.color = {.int32 = {0, 0, 0, 0}}};
#elif G17P_R32_UINT || G17P_RG16_UINT || G17P_RG8_UINT ||             \
   G17P_RG32_UINT || G17P_RGBA8_UINT || G17P_R8_UINT || G17P_RGBA16_UINT || \
   G17P_RGBA32_UINT || G17P_R16_UINT || G17P_A2B10G10R10_UINT
   const VkClearValue clear = {.color = {.uint32 = {0, 0, 0, 0}}};
#else
#if G17P_LOGIC_OP_XOR
   const VkClearValue clear = {
      .color = {.float32 = {1.0f / 255.0f * 0x55,
                            1.0f / 255.0f * 0xaa,
                            1.0f / 255.0f * 0x0f, 1.0f}},
   };
#else
   const VkClearValue clear = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};
#endif
#endif
   const VkRenderPassBeginInfo render_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {.extent = {IMAGE_WIDTH, IMAGE_HEIGHT}},
      .clearValueCount = G17P_RGBA8_MRT ? 2 : 1,
#if G17P_RGBA8_MRT
      .pClearValues = clears,
#else
      .pClearValues = &clear,
#endif
   };
#if G17P_SECONDARY_COMMAND_BUFFER
   const VkCommandBufferInheritanceInfo secondary_inheritance = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
      .renderPass = render_pass,
      .subpass = 0,
      .framebuffer = framebuffer,
   };
   const VkCommandBufferBeginInfo secondary_begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT |
               VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
      .pInheritanceInfo = &secondary_inheritance,
   };
#endif
   vkCmdBeginRenderPass(command_buffer, &render_begin,
#if G17P_SECONDARY_COMMAND_BUFFER
                        VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
#if G17P_SECONDARY_COMMAND_BUFFER_MULTI
   const VkDeviceSize secondary_vertex_offset = 0;
   for (unsigned quadrant = 0; quadrant < SECONDARY_QUADRANT_COUNT;
        ++quadrant) {
      VkCommandBuffer secondary = secondary_command_buffers[quadrant];
      check(vkBeginCommandBuffer(secondary, &secondary_begin),
            "vkBeginCommandBuffer(secondary)");
      vkCmdBindPipeline(secondary, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      vkCmdBindVertexBuffers(secondary, 0, 1, &vertex_buffer,
                             &secondary_vertex_offset);
      vkCmdPushConstants(secondary, pipeline_layout,
                         VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(secondary_quadrant_colors[quadrant]),
                         secondary_quadrant_colors[quadrant]);
      vkCmdDraw(secondary, SECONDARY_QUADRANT_VERTICES, 1,
                quadrant * SECONDARY_QUADRANT_VERTICES, 0);
      check(vkEndCommandBuffer(secondary), "vkEndCommandBuffer(secondary)");
   }

   /* vkQuake's R_SubmitContexts loop: one vkCmdExecuteCommands per secondary,
    * four separate consecutive calls inside this one render pass. Batching
    * them into a single commandBufferCount=4 call would not reproduce the
    * shape under test, so do not. */
   for (unsigned quadrant = 0; quadrant < SECONDARY_QUADRANT_COUNT;
        ++quadrant) {
      vkCmdExecuteCommands(command_buffer, 1,
                           &secondary_command_buffers[quadrant]);
   }
#else
   check(vkBeginCommandBuffer(secondary_command_buffer, &secondary_begin),
         "vkBeginCommandBuffer(secondary)");
   vkCmdBindPipeline(secondary_command_buffer,
                     VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   const VkDeviceSize secondary_vertex_offset = 0;
   vkCmdBindVertexBuffers(secondary_command_buffer, 0, 1, &vertex_buffer,
                          &secondary_vertex_offset);
   vkCmdDraw(secondary_command_buffer, 3, 1, 0, 0);
   check(vkEndCommandBuffer(secondary_command_buffer),
         "vkEndCommandBuffer(secondary)");
   vkCmdExecuteCommands(command_buffer, 1, &secondary_command_buffer);
#endif
#else
                        VK_SUBPASS_CONTENTS_INLINE);
#if !G17P_CUBE_SAMPLE
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
#endif
#if G17P_TWO_SAMPLERS_SPLIT_SETS
   /* Both sets bound in one call, first set 0, exactly as vkQuake binds its
    * diffuse and lightmap sets before a world draw. */
   vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline_layout, 0, 2, descriptor_sets, 0, NULL);
#elif G17P_TEXTURE_ORACLE && !G17P_TEXTURE_ALPHA_COMPOSITE
   vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
#endif
#if G17P_VERTEX_FORMAT
   const VkDeviceSize vertex_format_offset = VERTEX_FORMAT_BIND_OFFSET;
   vkCmdBindVertexBuffers(command_buffer, VERTEX_FORMAT_BINDING, 1,
                          &vertex_format.buffer, &vertex_format_offset);
#if G17P_DYNAMIC_VERTEX_INPUT
   /* Identical binding and attribute to the static pipeline above; the only
    * difference is that they arrive here instead of at pipeline creation. */
   const VkVertexInputBindingDescription2EXT dynamic_vi_binding = {
      .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
      .binding = VERTEX_FORMAT_BINDING,
      .stride = VERTEX_FORMAT_STRIDE,
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
      .divisor = 1,
   };
   const VkVertexInputAttributeDescription2EXT dynamic_vi_attribute = {
      .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
      .location = 0,
      .binding = VERTEX_FORMAT_BINDING,
      .format = VERTEX_INPUT_FORMAT,
      .offset = VERTEX_FORMAT_ATTRIBUTE_OFFSET,
   };
   vkCmdSetVertexInputEXT(command_buffer, 1, &dynamic_vi_binding, 1,
                          &dynamic_vi_attribute);
#endif
#elif G17P_INSTANCE_RATE
   const VkBuffer vertex_buffers[] = {vertex.buffer, instance_data.buffer};
   const VkDeviceSize vertex_offsets[] = {
      INSTANCE_VERTEX_BIND_OFFSET,
      INSTANCE_DATA_BIND_OFFSET,
   };
   vkCmdBindVertexBuffers(command_buffer, 0, 2, vertex_buffers,
                          vertex_offsets);
#else
#if G17P_TEXTURE_ORACLE
   const VkDeviceSize vertex_offset = TEXTURE_VERTEX_OFFSET;
#elif G17P_INDEX_RESTART
   const VkDeviceSize vertex_offset = RESTART_VERTEX_OFFSET;
#else
   const VkDeviceSize vertex_offset = 0;
#endif
   vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer,
                          &vertex_offset);
#endif
#if G17P_MULTI_VIEWPORT
#if G17P_MULTI_VIEWPORT_DYNAMIC_VIEWPORT
   vkCmdSetViewport(command_buffer, 0, 4, viewports);
#endif
#if G17P_MULTI_VIEWPORT_DYNAMIC_SCISSOR
   vkCmdSetScissor(command_buffer, 0, 4, scissors);
#endif
   vkCmdDraw(command_buffer, 3, 4, 0, 0);
#elif G17P_LAYER_OUTPUT
   vkCmdDraw(command_buffer, 3, IMAGE_LAYERS, 0, 0);
#elif G17P_VERTEX_FORMAT
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
#elif G17P_INSTANCE_RATE
   vkCmdDraw(command_buffer, 6, 2, 0, 3);
#elif G17P_TEXTURE_ALPHA_COMPOSITE
   vkCmdBindIndexBuffer(command_buffer, composite_index.buffer,
                        COMPOSITE_INDEX_BUFFER_OFFSET, VK_INDEX_TYPE_UINT16);
#if G17P_COMPOSITE_SINGLE_DRAW
   const VkRect2D alpha_scissor = {
      .extent = {IMAGE_WIDTH, IMAGE_HEIGHT},
   };
   const uint32_t dynamic_offset = 16 * COMPOSITE_UNIFORM_STRIDE;
   vkCmdSetScissor(command_buffer, 0, 1, &alpha_scissor);
   vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline_layout, 0, 1, &descriptor_set, 1,
                           &dynamic_offset);
   vkCmdDrawIndexed(command_buffer, 6, 1, COMPOSITE_FIRST_INDEX, 0, 0);
#elif G17P_COMPOSITE_NARROW_SINGLE
   /* Hold draw count at one while reproducing the physical 2-pixel scissor
    * width from the failing 32-wide opacity sweep. */
   const VkRect2D alpha_scissor = {
      .offset = {G17P_COMPOSITE_NARROW_X, G17P_COMPOSITE_NARROW_Y},
      .extent = {
         G17P_COMPOSITE_NARROW_WIDTH,
         G17P_COMPOSITE_NARROW_HEIGHT ? G17P_COMPOSITE_NARROW_HEIGHT
                                      : IMAGE_HEIGHT,
      },
   };
   const uint32_t dynamic_offset = COMPOSITE_UNIFORM_STRIDE;
   vkCmdSetScissor(command_buffer, 0, 1, &alpha_scissor);
   vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline_layout, 0, 1, &descriptor_set, 1,
                           &dynamic_offset);
#if G17P_COMPOSITE_FULLSCREEN_TRIANGLE
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
#else
   vkCmdDrawIndexed(command_buffer, 6, 1, COMPOSITE_FIRST_INDEX, 0, 0);
#endif
#elif G17P_COMPOSITE_REPEATED_FULL
   /* Hold the physical scissor at the full target while retaining all 17
    * dynamic-state/bind/draw transitions. The first 16 draws have zero
    * opacity; only the final opaque draw contributes to the exact oracle. */
   const VkRect2D alpha_scissor = {
      .extent = {IMAGE_WIDTH, IMAGE_HEIGHT},
   };
   for (uint32_t step = 0; step < COMPOSITE_OPACITY_COUNT; ++step) {
      const uint32_t dynamic_offset =
         (step + 1 == COMPOSITE_OPACITY_COUNT ? 16 : 0) *
         COMPOSITE_UNIFORM_STRIDE;
      vkCmdSetScissor(command_buffer, 0, 1, &alpha_scissor);
      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipeline_layout, 0, 1, &descriptor_set, 1,
                              &dynamic_offset);
      vkCmdDrawIndexed(command_buffer, 6, 1, COMPOSITE_FIRST_INDEX, 0, 0);
   }
#elif G17P_COMPOSITE_NARROW_TWO || G17P_COMPOSITE_NARROW_TWO_CHANGED
   /* Select the same narrow record at indices 0 and 1, or change width 1 to
    * width 2 while advancing to index 1. The transparent first draw makes
    * both cases share one exact final image. */
   for (uint32_t step = 0; step < 2; ++step) {
      const VkRect2D alpha_scissor = {
         .extent = {
            G17P_COMPOSITE_NARROW_TWO_CHANGED && step == 0 ? 1 : 2,
            IMAGE_HEIGHT,
         },
      };
      const uint32_t dynamic_offset = step * COMPOSITE_UNIFORM_STRIDE;
      vkCmdSetScissor(command_buffer, 0, 1, &alpha_scissor);
      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipeline_layout, 0, 1, &descriptor_set, 1,
                              &dynamic_offset);
      vkCmdDrawIndexed(command_buffer, 6, 1, COMPOSITE_FIRST_INDEX, 0, 0);
   }
#else
   for (uint32_t step = 0; step < COMPOSITE_OPACITY_COUNT; ++step) {
      const uint32_t min_x = step * IMAGE_WIDTH / COMPOSITE_OPACITY_COUNT;
      const uint32_t max_x =
         (step + 1) * IMAGE_WIDTH / COMPOSITE_OPACITY_COUNT;
      const VkRect2D alpha_scissor = {
         .offset = {min_x, 0},
         .extent = {max_x - min_x, IMAGE_HEIGHT},
      };
      const uint32_t dynamic_offset = step * COMPOSITE_UNIFORM_STRIDE;
      vkCmdSetScissor(command_buffer, 0, 1, &alpha_scissor);
      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipeline_layout, 0, 1, &descriptor_set, 1,
                              &dynamic_offset);
      vkCmdDrawIndexed(command_buffer, 6, 1, COMPOSITE_FIRST_INDEX, 0, 0);
   }
#endif
#elif G17P_INDEX_RESTART
   vkCmdBindIndexBuffer(command_buffer, restart_index.buffer,
                        RESTART_INDEX_BUFFER_OFFSET, VK_INDEX_TYPE_UINT16);
   vkCmdDrawIndexed(command_buffer, 7, 1, RESTART_FIRST_INDEX, 0, 0);
#elif G17P_INDIRECT_DRAW == 1
   vkCmdDrawIndirect(command_buffer, indirect.buffer, INDIRECT_COMMAND_OFFSET,
                     2, INDIRECT_COMMAND_STRIDE);
#elif G17P_INDIRECT_DRAW == 2 || G17P_ROBUST_INDEXED
   vkCmdBindIndexBuffer(command_buffer, index.buffer, INDEX_BUFFER_OFFSET,
                        VK_INDEX_TYPE_UINT16);
   vkCmdDrawIndexedIndirect(command_buffer, indirect.buffer,
                            INDIRECT_COMMAND_OFFSET, 2,
                            INDIRECT_COMMAND_STRIDE);
#elif G17P_INDIRECT_DRAW == 3
   vkCmdDrawIndirectCount(command_buffer, indirect.buffer,
                          INDIRECT_COMMAND_OFFSET, indirect_count.buffer,
                          INDIRECT_COUNT_OFFSET, 0, INDIRECT_COMMAND_STRIDE);
   vkCmdDrawIndirectCount(command_buffer, indirect.buffer,
                          INDIRECT_COMMAND_OFFSET, indirect_count.buffer,
                          INDIRECT_COUNT_OFFSET, 3, INDIRECT_COMMAND_STRIDE);
#elif G17P_INDIRECT_DRAW == 4
   vkCmdBindIndexBuffer(command_buffer, index.buffer, INDEX_BUFFER_OFFSET,
                        VK_INDEX_TYPE_UINT16);
   vkCmdDrawIndexedIndirectCount(
      command_buffer, indirect.buffer, INDIRECT_COMMAND_OFFSET,
      indirect_count.buffer, INDIRECT_COUNT_OFFSET, 0,
      INDIRECT_COMMAND_STRIDE);
   vkCmdDrawIndexedIndirectCount(
      command_buffer, indirect.buffer, INDIRECT_COMMAND_OFFSET,
      indirect_count.buffer, INDIRECT_COUNT_OFFSET, 3,
      INDIRECT_COMMAND_STRIDE);
#elif G17P_CUBE_SAMPLE
   for (uint32_t face = 0; face < 6; ++face) {
      if (cube_face_diagnostic >= 0 && cube_face_diagnostic != (int)face)
         continue;
      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        cube_pipelines[face]);
      vkCmdDraw(command_buffer, 3, 1, 0, 0);
   }
#elif G17P_BRESENHAM_LINES
   vkCmdDraw(command_buffer, 2, 1, 0, 0);
#elif G17P_MSAA_COVERAGE == 1
   const VkSampleMask dynamic_sample_mask = 0x5;
   vkCmdSetSampleMaskEXT(command_buffer, VK_SAMPLE_COUNT_4_BIT,
                         &dynamic_sample_mask);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
#elif G17P_MSAA_COVERAGE == 3
   const VkRect2D mask_scissors[2] = {
      {.extent = {IMAGE_WIDTH / 2, IMAGE_HEIGHT}},
      {
         .offset = {IMAGE_WIDTH / 2, 0},
         .extent = {IMAGE_WIDTH / 2, IMAGE_HEIGHT},
      },
   };
   /* Change the mask between two otherwise identical draws: an all-samples
    * mask must retain the fragment, an empty mask must reject it entirely.
    * A pipeline that ignored the dynamic mask would paint both halves. */
   VkSampleMask dynamic_sample_mask = 0xf;
   vkCmdSetSampleMaskEXT(command_buffer, VK_SAMPLE_COUNT_4_BIT,
                         &dynamic_sample_mask);
   vkCmdSetScissor(command_buffer, 0, 1, &mask_scissors[0]);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
   dynamic_sample_mask = 0x0;
   vkCmdSetSampleMaskEXT(command_buffer, VK_SAMPLE_COUNT_4_BIT,
                         &dynamic_sample_mask);
   vkCmdSetScissor(command_buffer, 0, 1, &mask_scissors[1]);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
#elif G17P_MSAA_COVERAGE == 2
   const VkRect2D alpha_scissors[2] = {
      {.extent = {IMAGE_WIDTH / 2, IMAGE_HEIGHT}},
      {
         .offset = {IMAGE_WIDTH / 2, 0},
         .extent = {IMAGE_WIDTH / 2, IMAGE_HEIGHT},
      },
   };
   vkCmdSetAlphaToCoverageEnableEXT(command_buffer, VK_FALSE);
   vkCmdSetScissor(command_buffer, 0, 1, &alpha_scissors[0]);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
   vkCmdSetAlphaToCoverageEnableEXT(command_buffer, VK_TRUE);
   vkCmdSetScissor(command_buffer, 0, 1, &alpha_scissors[1]);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
#elif G17P_DYNAMIC_BLEND
   const VkRect2D dynamic_scissors[2] = {
      {.extent = {IMAGE_WIDTH / 2, IMAGE_HEIGHT}},
      {
         .offset = {IMAGE_WIDTH / 2, 0},
         .extent = {IMAGE_WIDTH / 2, IMAGE_HEIGHT},
      },
   };
   const VkColorBlendEquationEXT reverse_subtract = {
      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .colorBlendOp = VK_BLEND_OP_REVERSE_SUBTRACT,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .alphaBlendOp = VK_BLEND_OP_ADD,
   };
   const VkColorBlendEquationEXT overwrite = {
      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp = VK_BLEND_OP_ADD,
   };
   const VkBool32 enables[2] = {VK_TRUE, VK_FALSE};
   const VkColorComponentFlags masks[2] = {
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT,
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkColorBlendEquationEXT equations[2] = {
      reverse_subtract, overwrite,
   };
   for (unsigned i = 0; i < 2; ++i) {
      vkCmdSetScissor(command_buffer, 0, 1, &dynamic_scissors[i]);
      vkCmdSetColorBlendEnableEXT(command_buffer, 0, 1, &enables[i]);
      vkCmdSetColorBlendEquationEXT(command_buffer, 0, 1, &equations[i]);
      vkCmdSetColorWriteMaskEXT(command_buffer, 0, 1, &masks[i]);
      vkCmdDraw(command_buffer, 3, 1, 0, 0);
   }
#else
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
#endif
#endif
   vkCmdEndRenderPass(command_buffer);

#if G17P_RGBA8_MRT
   const VkImageMemoryBarrier mrt_barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1,
         .layerCount = 1,
      },
   };
   vkCmdPipelineBarrier(command_buffer,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                        0, NULL, 0, NULL, 1, &mrt_barrier);

   const VkRenderPassBeginInfo load_render_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = load_render_pass,
      .framebuffer = load_framebuffer,
      .renderArea = {.extent = {IMAGE_WIDTH, IMAGE_HEIGHT}},
   };
   vkCmdBeginRenderPass(command_buffer, &load_render_begin,
                        VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                     additive_pipeline);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
   vkCmdEndRenderPass(command_buffer);
#endif

#if !G17P_HOST_IMAGE_COPY
   const VkBufferImageCopy copy = {
      .bufferOffset = READBACK_GUARD_BYTES,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .layerCount = IMAGE_LAYERS,
      },
      .imageExtent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
   };
   if (!record_only) {
      vkCmdCopyImageToBuffer(command_buffer, image,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
                             &copy);
#if G17P_RGBA8_MRT
      VkBufferImageCopy copy1 = copy;
      copy1.bufferOffset = READBACK_IMAGE1_OFFSET;
      vkCmdCopyImageToBuffer(command_buffer, image1,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
                             &copy1);
#endif
   }
#endif
   check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

   if (record_only && !G17P_MULTIVIEW && !G17P_LAYER_OUTPUT &&
       !G17P_MULTI_VIEWPORT && !G17P_SHADER_QUAD_CONTROL &&
       !G17P_SECONDARY_COMMAND_BUFFER && !G17P_ROBUST_INDEXED) {
      printf(PROBE_NAME ": PASS — feature advertised, shaders compiled, "
                        "vkEndCommandBuffer succeeded\n");
      return EXIT_SUCCESS;
   }

   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkFence fence;
   check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");
   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
   };
   check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
#if G17P_TEXTURE_ORACLE || G17P_VERTEX_FORMAT || G17P_INSTANCE_RATE ||   \
   G17P_MULTIVIEW || G17P_LAYER_OUTPUT || G17P_MULTI_VIEWPORT ||        \
   G17P_INDEX_RESTART ||                                                \
   G17P_INDIRECT_DRAW || G17P_F16_IO || G17P_VERTEX_OUTPUT_68 ||        \
   G17P_BRESENHAM_LINES || G17P_LOGIC_OP_XOR || G17P_BLEND_CONSTANT ||  \
   G17P_RGBA8_MRT || G17P_DYNAMIC_BLEND || G17P_ADVANCED_BLEND ||      \
   G17P_RASTER_DISCARD || G17P_SHADER_QUAD_CONTROL ||                   \
   G17P_RG16_UNORM ||                                                    \
   G17P_RGBA16_UNORM ||                                                \
   G17P_RG16_SNORM || G17P_R32_UINT || G17P_R32_SFLOAT ||             \
   G17P_R32_SINT || G17P_RG16_UINT || G17P_RG16_SINT ||               \
   G17P_RG32_SINT || G17P_RG32_SFLOAT || G17P_RG8_UINT ||             \
   G17P_RG8_SINT || G17P_RG32_UINT || G17P_RGBA8_UINT ||              \
   G17P_RGBA8_SINT || G17P_RGBA8_SNORM || G17P_BGRA8_SNORM ||        \
   G17P_BGRA8_UNORM || G17P_BGRA8_SRGB ||                             \
   G17P_RG8_SNORM ||                                                   \
   G17P_R8_UINT || G17P_R8_UNORM || G17P_R8_SNORM || G17P_R8_SINT || \
   G17P_RGBA16_SNORM || G17P_R16_SNORM || G17P_R16_UNORM ||          \
   G17P_RGBA16_UINT || G17P_RGBA16_SINT || G17P_RGBA32_UINT ||        \
   G17P_RGBA32_SINT || G17P_R16_UINT || G17P_R16_SINT || G17P_R16_SFLOAT || \
   G17P_RG16_SFLOAT || G17P_RGBA16_SFLOAT || G17P_A2B10G10R10_UINT || \
   G17P_A2R10G10B10_UNORM || G17P_RGBA32_SFLOAT
   const VkResult waited =
      vkWaitForFences(device, 1, &fence, true, UINT64_C(3000000000));
   if (waited == VK_TIMEOUT)
      die("vkWaitForFences timed out after three seconds");
   check(waited, "vkWaitForFences");
#elif G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES > 1
   const VkResult waited =
      vkWaitForFences(device, 1, &fence, true, UINT64_C(5000000000));
   if (waited == VK_TIMEOUT)
      die("vkWaitForFences timed out after five seconds");
   check(waited, "vkWaitForFences");
#else
   check(vkWaitForFences(device, 1, &fence, true, UINT64_MAX),
         "vkWaitForFences");
#endif

#if G17P_MULTI_VIEWPORT
   if (record_only) {
      printf(PROBE_NAME ": PASS — feature advertised, shader compiled, "
                        "and four-viewport command stream submitted\n");
      return EXIT_SUCCESS;
   }
#elif G17P_LAYER_OUTPUT
   if (record_only) {
      printf(PROBE_NAME ": PASS — feature advertised, shader compiled, "
                        "and two-instance/layer command stream submitted\n");
      return EXIT_SUCCESS;
   }
#elif G17P_MULTIVIEW
   if (record_only) {
      printf(PROBE_NAME ": PASS — feature advertised, shaders compiled, "
                        "and two-view command stream submitted\n");
      return EXIT_SUCCESS;
   }
#elif G17P_SHADER_QUAD_CONTROL
   if (record_only) {
      printf(PROBE_NAME ": PASS — feature advertised, divergent quad shader "
                        "compiled, and command stream submitted\n");
      return EXIT_SUCCESS;
   }
#elif G17P_SECONDARY_COMMAND_BUFFER_MULTI
   if (record_only) {
      printf(PROBE_NAME ": RECORD PASS — four inherited secondary draws "
                        "replayed by four consecutive vkCmdExecuteCommands "
                        "calls into a validated G17P command stream\n");
      return EXIT_SUCCESS;
   }
#elif G17P_SECONDARY_COMMAND_BUFFER
   if (record_only) {
      printf(PROBE_NAME ": RECORD PASS — inherited secondary draw replayed "
                        "into a validated G17P command stream\n");
      return EXIT_SUCCESS;
   }
#elif G17P_ROBUST_INDEXED
   if (record_only) {
      printf(PROBE_NAME ": RECORD PASS — two robust indexed indirect draws "
                        "submitted with fresh and reused HAL200 draw state\n");
      return EXIT_SUCCESS;
   }
#endif

   if (submit_only) {
      printf(PROBE_NAME ": SUBMIT PASS — generated-mip command stream "
                        "submitted and completed\n");
      return EXIT_SUCCESS;
   }

#if G17P_HOST_IMAGE_COPY
   const VkImageToMemoryCopyEXT host_copy_region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY_EXT,
      .pHostPointer = readback_map,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .layerCount = 1,
      },
      .imageExtent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
   };
   const VkCopyImageToMemoryInfoEXT host_copy = {
      .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO_EXT,
      .srcImage = image,
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .regionCount = 1,
      .pRegions = &host_copy_region,
   };
   check(vkCopyImageToMemoryEXT(device, &host_copy),
         "vkCopyImageToMemoryEXT");
#endif

#if G17P_COMPOSITE_GENERATE_MIPS
   if (!verify_generated_mip1(generated_mip_readback.map))
      return EXIT_FAILURE;
#if G17P_COMPOSITE_GENERATE_MIPS_ODD
   printf(PROBE_NAME ": PASS — odd mip1 matches the native linear-filter oracle\n");
   return EXIT_SUCCESS;
#endif
#endif
   if (!verify_readback(readback_map))
      return EXIT_FAILURE;

#if G17P_VERTEX_FORMAT
   if (!verify_upload(&vertex_format, vertex_format_expected,
                      "vertex-format input"))
      return EXIT_FAILURE;
#endif
#if G17P_INSTANCE_RATE
   if (!verify_upload(&vertex, vertex_expected, "vertex input") ||
       !verify_upload(&instance_data, instance_expected, "instance input"))
      return EXIT_FAILURE;
#endif
#if G17P_INDIRECT_DRAW
   if (!verify_upload(&indirect, indirect_expected, "indirect input"))
      return EXIT_FAILURE;
#if G17P_INDIRECT_COUNT
   if (!verify_upload(&indirect_count, indirect_count_expected,
                      "indirect count input"))
      return EXIT_FAILURE;
#endif
#if G17P_INDIRECT_INDEXED
   if (!verify_upload(&index, index_expected, "index input"))
      return EXIT_FAILURE;
#endif
#endif
#if G17P_INDEX_RESTART
   if (!verify_upload(&vertex, vertex_expected, "vertex input") ||
       !verify_upload(&restart_index, restart_index_expected, "index input"))
      return EXIT_FAILURE;
#endif
#if G17P_TEXTURE_ORACLE
   if (!verify_upload(&texture_vertex, texture_vertex_expected,
                      "vertex input")
#if !G17P_SHADOW_COMPARE
       ||
       !verify_upload(&texture_upload, texture_upload_expected,
                      "texture upload")
#endif
   )
      return EXIT_FAILURE;
#endif
#if G17P_TEXTURE_ALPHA_COMPOSITE
   if (!verify_upload(&composite_index, composite_index_expected,
                      "composite index input") ||
       !verify_upload(&composite_uniform, composite_uniform_expected,
                      "composite uniform input"))
      return EXIT_FAILURE;
#endif

#if G17P_TEXTURE_ORACLE || G17P_VERTEX_FORMAT || G17P_INSTANCE_RATE ||     \
   G17P_MULTIVIEW || G17P_LAYER_OUTPUT || G17P_MULTI_VIEWPORT ||           \
   G17P_INDEX_RESTART ||                                                  \
   G17P_INDIRECT_DRAW ||                                                 \
   G17P_MSAA_SAMPLES > 1 || G17P_F16_IO || G17P_VERTEX_OUTPUT_68 ||      \
   G17P_BRESENHAM_LINES ||                                                \
   G17P_LOGIC_OP_XOR || G17P_BLEND_CONSTANT || G17P_RGBA8_MRT ||        \
   G17P_DYNAMIC_BLEND || G17P_ADVANCED_BLEND || G17P_RASTER_DISCARD || \
   G17P_SHADER_QUAD_CONTROL || G17P_RG16_UNORM ||                         \
   G17P_RGBA16_UNORM || G17P_RG16_SNORM || G17P_R32_UINT ||             \
   G17P_R32_SFLOAT || G17P_R32_SINT || G17P_RG16_UINT ||                \
   G17P_RG16_SINT || G17P_RG32_SINT || G17P_RG32_SFLOAT ||            \
   G17P_RG8_UINT || G17P_RG8_SINT || G17P_RG32_UINT ||                \
   G17P_RGBA8_UINT || G17P_RGBA8_SINT || G17P_RGBA8_SNORM ||           \
   G17P_BGRA8_SNORM ||                                                  \
   G17P_RG8_SNORM || G17P_R8_UINT || G17P_R8_SNORM || G17P_R8_SINT || \
   G17P_RGBA16_SNORM || G17P_R16_SNORM || G17P_R16_UNORM ||            \
   G17P_RGBA16_UINT || G17P_RGBA16_SINT || G17P_RGBA32_UINT ||         \
   G17P_RGBA32_SINT || G17P_R16_UINT || G17P_R16_SINT || G17P_R16_SFLOAT || \
   G17P_RG16_SFLOAT || G17P_RGBA16_SFLOAT || G17P_A2B10G10R10_UINT || \
   G17P_A2R10G10B10_UNORM || G17P_RGBA32_SFLOAT
   sleep(3);
#if G17P_COMPOSITE_GENERATE_MIPS
   if (!verify_generated_mip1(generated_mip_readback.map))
      return EXIT_FAILURE;
#endif
   if (!verify_readback(readback_map))
      return EXIT_FAILURE;
#if G17P_VERTEX_FORMAT
   if (!verify_upload(&vertex_format, vertex_format_expected,
                      "vertex-format input"))
      return EXIT_FAILURE;
#endif
#if G17P_INSTANCE_RATE
   if (!verify_upload(&vertex, vertex_expected, "vertex input") ||
       !verify_upload(&instance_data, instance_expected, "instance input"))
      return EXIT_FAILURE;
#endif
#if G17P_INDIRECT_DRAW
   if (!verify_upload(&indirect, indirect_expected, "indirect input"))
      return EXIT_FAILURE;
#if G17P_INDIRECT_COUNT
   if (!verify_upload(&indirect_count, indirect_count_expected,
                      "indirect count input"))
      return EXIT_FAILURE;
#endif
#if G17P_INDIRECT_INDEXED
   if (!verify_upload(&index, index_expected, "index input"))
      return EXIT_FAILURE;
#endif
#endif
#if G17P_INDEX_RESTART
   if (!verify_upload(&vertex, vertex_expected, "vertex input") ||
       !verify_upload(&restart_index, restart_index_expected, "index input"))
      return EXIT_FAILURE;
#endif
#if G17P_TEXTURE_ORACLE
   if (!verify_upload(&texture_vertex, texture_vertex_expected,
                      "vertex input")
#if !G17P_SHADOW_COMPARE
       ||
       !verify_upload(&texture_upload, texture_upload_expected,
                      "texture upload")
#endif
   )
      return EXIT_FAILURE;
#endif
#if G17P_TEXTURE_ALPHA_COMPOSITE
   if (!verify_upload(&composite_index, composite_index_expected,
                      "composite index input") ||
       !verify_upload(&composite_uniform, composite_uniform_expected,
                      "composite uniform input"))
      return EXIT_FAILURE;
#endif
#endif
#if G17P_RGBA8_MRT
   const uint64_t checksum =
      fnv1a64(readback_map + READBACK_GUARD_BYTES, IMAGE_BYTES);
   const uint64_t checksum1 =
      fnv1a64(readback_map + READBACK_IMAGE1_OFFSET, IMAGE_BYTES);
#if G17P_STK_R11G11B10_MRT
   printf(PROBE_NAME ": PASS — two distinct B10G11R11 UFLOAT attachments "
                     "preserved exact clears outside the scissor; RT0 "
                     "STORE/LOAD/add and RT1 independent store remained "
                     "exact inside it immediately/3s; three guards unchanged, "
                     "checksums %016" PRIx64 "/%016" PRIx64 "\n",
          checksum, checksum1);
#else
   printf(PROBE_NAME ": PASS — two distinct RGBA8 attachments preserved "
                     "exact clears outside the scissor; RT0 STORE/LOAD/add "
                     "and RT1 independent store remained exact inside it "
                     "immediately/3s; "
                     "three guards unchanged, checksums %016" PRIx64
                     "/%016" PRIx64 "\n", checksum, checksum1);
#endif
#else
   const uint64_t checksum =
      fnv1a64(readback_map + READBACK_GUARD_BYTES, READBACK_PAYLOAD_BYTES);
#if G17P_TEXTURE_MIN_LOD
   printf(PROBE_NAME ": PASS op=%u minLod=%.0f; every pixel and guard exact, "
                     "checksum %016" PRIx64 "\n",
          G17P_TEXTURE_MIN_LOD, view_min_lod, checksum);
#elif G17P_MULTI_VIEWPORT
   printf(PROBE_NAME ": PASS — four instances selected exact quadrant "
                     "viewports 0 through 3 immediately/3s; all guards "
                     "unchanged, checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_LAYER_OUTPUT
   printf(PROBE_NAME ": PASS — two instances selected exact layers 0 and 1 "
                     "immediately/3s; all readback guards unchanged, "
                     "checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_VERTEX_OUTPUT_68
   printf(PROBE_NAME ": PASS — position plus seventeen float4 user outputs "
                     "(68 user/72 total UVS words) produced the exact image "
                     "immediately/3s; all readback guards unchanged, "
                     "checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_SHADER_QUAD_CONTROL
   printf(PROBE_NAME ": PASS — divergent odd-lane loop retained exact "
                     "horizontal-quad partner values and quad derivatives "
                     "immediately/3s; guards unchanged, checksum %016" PRIx64
                     "\n",
          checksum);
#elif G17P_MULTIVIEW
   printf(PROBE_NAME ": PASS — both view 0/layer 0 and view 1/layer 1 "
                     "contain the exact full-frame image immediately/3s; "
                     "all readback guards unchanged, checksum %016" PRIx64
                     "\n",
          checksum);
#elif G17P_IMPLICIT_LOD
   printf(PROBE_NAME ": PASS — implicit derivative LOD selected exact mip 1 "
                     "immediately/3s, distinct red/green/blue mips and all "
                     "source/readback guards unchanged, checksum %016" PRIx64
                     "\n",
          checksum);
#elif G17P_TEXTURE_ALPHA_COMPOSITE
   printf(PROBE_NAME ": PASS — dynamically indexed descriptor 1 and UBO "
                     "offsets selected the checker mip 1 and all 17 opacity "
                     "steps; quarter-texel linear RGBA sampling, clamp edges, "
                     "two-triangle diagonal coverage and premultiplied blend "
                     "were exact "
                     "immediately/3s; all source/readback guards unchanged, "
                     "checksum %016" PRIx64 "\n", checksum);
#elif G17P_TWO_SAMPLERS
   {
      static const uint8_t wanted[4] = G17P_TWO_SAMPLERS_EXPECTED;
      const uint8_t *centre =
         readback_map + READBACK_GUARD_BYTES +
         ((IMAGE_HEIGHT / 2) * IMAGE_WIDTH + IMAGE_WIDTH / 2) *
            IMAGE_PIXEL_BYTES;
      printf(PROBE_NAME ": PASS — two distinct sampled textures in %s "
                        "multiplied to their exact product immediately/3s; "
                        "centre pixel RGBA %u %u %u %u against expected "
                        "%u %u %u %u +/- %u; all source/readback guards "
                        "unchanged, checksum %016" PRIx64 "\n",
             G17P_TWO_SAMPLERS_SPLIT_SETS ? "two descriptor sets"
                                          : "one descriptor set",
             centre[0], centre[1], centre[2], centre[3], wanted[0], wanted[1],
             wanted[2], wanted[3], G17P_TWO_SAMPLERS_TOLERANCE, checksum);
   }
#elif G17P_EXPLICIT_GRADIENT
   printf(PROBE_NAME ": PASS — explicit gradients selected exact mip 1 "
                     "immediately/3s, distinct red/green/blue mips and all "
                     "source/readback guards unchanged, checksum %016" PRIx64
                     "\n",
          checksum);
#elif G17P_TEXTURE_GATHER
   printf(PROBE_NAME ": PASS — textureGather red-component footprint and "
                     "lane order exact immediately/3s, all source/readback "
                     "guards unchanged, checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_SATURATE_TEXTURE
   printf(PROBE_NAME ": PASS — explicitly lowered saturated texture multiply "
                     "produced exact lower, upper, and interior lanes "
                     "immediately/3s; all source/readback guards unchanged, "
                     "checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_SHADOW_COMPARE
   printf(PROBE_NAME ": PASS — two depth references split the exact "
                     "white/black halves immediately/3s; all source/readback guards "
                     "unchanged, checksum %016" PRIx64 "\n",
                     checksum);
#elif G17P_TEXEL_FETCH_LOD
   printf(PROBE_NAME ": PASS — texelFetch selected exact mip 2 through "
                     "the native immediate-level path immediately/3s; "
                     "all source/readback guards unchanged, checksum %016" PRIx64
                     "\n", checksum);
#elif G17P_CUBE_SAMPLE
   if (cube_face_diagnostic >= 0) {
      printf(PROBE_NAME ": PASS — cube face %d sampled its exact RGBA color "
                        "in one draw immediately/3s; all other stripes stayed "
                        "clear and all guards unchanged, checksum %016" PRIx64
                        "\n", cube_face_diagnostic, checksum);
   } else {
      printf(PROBE_NAME ": PASS — all six cube faces sampled their exact "
                        "distinct RGBA colors immediately/3s; six-layer upload "
                        "and all source/readback guards unchanged, checksum %016" PRIx64
                        "\n", checksum);
   }
#elif G17P_TEXTURE_3D
   printf(PROBE_NAME ": PASS — explicit LOD1 3D sampling selected exact "
                     "blue slice 1 from two distinct mip slices immediately/3s; "
                     "all source/readback guards unchanged, checksum %016" PRIx64
                     "\n", checksum);
#elif G17P_YCBCR
   printf(PROBE_NAME ": PASS — two-plane 4:2:0 upload and immutable "
                     "RGB-identity conversion produced exact Cr/Y/Cb output "
                     "immediately/3s; all source/readback guards unchanged, "
                     "checksum %016" PRIx64 "\n", checksum);
#elif G17P_VERTEX_FORMAT == 1
   printf(PROBE_NAME ": PASS — RGBA8 UNORM vertex conversion exact "
                     "immediately/3s, binding 3, offset 64, stride 13, "
                     "attribute offset 5, all source/readback guards "
                     "unchanged, checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_VERTEX_FORMAT == 2
   printf(PROBE_NAME ": PASS — RG16 SINT integer vertex input exact "
                     "immediately/3s, binding 3, offset 64, stride 14, "
                     "attribute offset 6, all source/readback guards "
                     "unchanged, checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_VERTEX_FORMAT == 3
   printf(PROBE_NAME ": PASS — RG16 SFLOAT vertex conversion exact "
                     "immediately/3s, binding 3, offset 64, stride 10, "
                     "attribute offset 6, all source/readback guards "
                     "unchanged, checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_INSTANCE_RATE
   printf(PROBE_NAME ": PASS — two distinct instance-rate quads exact "
                     "immediately/3s, firstInstance 3, padded binding/stride/"
                     "attribute offsets, all source/readback guards unchanged, "
                     "checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_INDEX_RESTART
   printf(PROBE_NAME ": PASS — triangle-strip primitive restart exact "
                     "immediately/3s, firstIndex 31 crosses byte 128, "
                     "all vertex/index/readback guards unchanged, checksum "
                     "%016" PRIx64 "\n",
          checksum);
#elif G17P_INDIRECT_DRAW == 1
   printf(PROBE_NAME ": PASS — vkCmdDrawIndirect exact immediately/3s, "
                     "two records/stride, nonzero command offset/firstVertex/"
                     "firstInstance, "
                     "256 guard bytes unchanged, checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_INDIRECT_DRAW == 2
   printf(PROBE_NAME ": PASS — vkCmdDrawIndexedIndirect exact immediately/3s, "
                     "two records/stride, nonzero command/index offsets, "
                     "firstIndex/baseVertex/firstInstance, 384 guard bytes "
                     "unchanged, checksum "
                     "%016" PRIx64 "\n",
          checksum);
#elif G17P_ROBUST_INDEXED
   printf(PROBE_NAME ": PASS — robust vkCmdDrawIndexedIndirect exact "
                     "immediately/3s, two records exercise fresh/reused "
                     "HAL200 draw state, all command/index/readback guards "
                     "unchanged, checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_INDIRECT_DRAW == 3
   printf(PROBE_NAME ": PASS — vkCmdDrawIndirectCount exact immediately/3s, "
                     "count-buffer and command offsets, max-count "
                     "predication, stride/firstVertex/firstInstance, all "
                     "guards unchanged, checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_INDIRECT_DRAW == 4
   printf(PROBE_NAME ": PASS — vkCmdDrawIndexedIndirectCount exact "
                     "immediately/3s, count-buffer/command/index offsets, "
                     "max-count predication, firstIndex/baseVertex/"
                     "firstInstance, all guards unchanged, checksum %016"
                     PRIx64 "\n",
          checksum);
#elif G17P_A2R10G10B10_UNORM
   printf(PROBE_NAME ": PASS — %ux%u A2R10G10B10 UNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA32_SFLOAT
   printf(PROBE_NAME ": PASS — %ux%u RGBA32 SFLOAT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_A2B10G10R10_UINT
   printf(PROBE_NAME ": PASS — %ux%u A2B10G10R10 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA16_SFLOAT
   printf(PROBE_NAME ": PASS — %ux%u RGBA16 SFLOAT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG16_SFLOAT
   printf(PROBE_NAME ": PASS — %ux%u RG16 SFLOAT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R16_SFLOAT
   printf(PROBE_NAME ": PASS — %ux%u R16 SFLOAT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R16_SINT
   printf(PROBE_NAME ": PASS — %ux%u R16 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R16_UINT
   printf(PROBE_NAME ": PASS — %ux%u R16 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA32_SINT
   printf(PROBE_NAME ": PASS — %ux%u RGBA32 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA32_UINT
   printf(PROBE_NAME ": PASS — %ux%u RGBA32 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA16_SINT
   printf(PROBE_NAME ": PASS — %ux%u RGBA16 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_BRESENHAM_LINES
   printf(PROBE_NAME ": PASS — 32x32 exact 3px Bresenham line immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_LOGIC_OP_XOR
   printf(PROBE_NAME ": PASS — exact RGBA8 XOR with retained clear "
                     "immediately/3s, guards unchanged, checksum %016" PRIx64
                     "\n", checksum);
#elif G17P_BLEND_CONSTANT
   printf(PROBE_NAME ": PASS — exact independent constant-color/alpha "
                     "blend immediately/3s, guards unchanged, checksum %016"
                     PRIx64 "\n", checksum);
#elif G17P_DYNAMIC_BLEND
   printf(PROBE_NAME ": PASS — dynamic blend enable/equation/write-mask "
                     "updates produced exact split output immediately/3s, "
                     "guards unchanged, checksum %016" PRIx64 "\n",
          checksum);
#elif G17P_SAMPLE_RATE_SHADING
   printf(PROBE_NAME ": PASS — four distinct SampleID colours averaged to "
                     "exact RGBA8 immediately/3s; all guards unchanged, "
                     "checksum %016" PRIx64 "\n", checksum);
#elif G17P_MSAA_COVERAGE == 1
   printf(PROBE_NAME ": PASS — dynamic sample mask 0b0101 retained exactly "
                     "samples 0/2 through 4x resolve immediately/3s; all "
                     "guards unchanged, checksum %016" PRIx64 "\n", checksum);
#elif G17P_MSAA_COVERAGE == 3
   printf(PROBE_NAME ": PASS — dynamic sample mask 0b1111 retained and 0b0000 "
                     "rejected an otherwise identical 4x draw "
                     "immediately/3s; all guards unchanged, checksum %016" PRIx64
                     "\n", checksum);
#elif G17P_MSAA_COVERAGE == 2
   printf(PROBE_NAME ": PASS — dynamic alpha-to-coverage toggled an "
                     "alpha-zero draw from fully retained to fully rejected "
                     "immediately/3s; all guards unchanged, checksum %016" PRIx64
                     "\n", checksum);
#elif G17P_ADVANCED_BLEND
   printf(PROBE_NAME ": PASS — coherent conjoint MULTIPLY produced exact "
                     "nonpremultiplied RGBA8 output immediately/3s, guards "
                     "unchanged, checksum %016" PRIx64 "\n", checksum);
#elif G17P_RASTER_DISCARD
   printf(PROBE_NAME ": PASS — rasterizer discard suppressed the draw while "
                     "the exact clear survived immediately/3s, guards "
                     "unchanged, checksum %016" PRIx64 "\n", checksum);
#elif G17P_RGBA16_UINT
   printf(PROBE_NAME ": PASS — %ux%u RGBA16 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R16_UNORM
   printf(PROBE_NAME ": PASS — %ux%u R16 UNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R16_SNORM
   printf(PROBE_NAME ": PASS — %ux%u R16 SNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA16_SNORM
   printf(PROBE_NAME ": PASS — %ux%u RGBA16 SNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R8_SINT
   printf(PROBE_NAME ": PASS — %ux%u R8 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R8_UNORM
   printf(PROBE_NAME ": PASS — %ux%u R8 UNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R8_SNORM
   printf(PROBE_NAME ": PASS — %ux%u R8 SNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R8_UINT
   printf(PROBE_NAME ": PASS — %ux%u R8 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG8_SNORM
   printf(PROBE_NAME ": PASS — %ux%u RG8 SNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_BGRA8_SNORM
   printf(PROBE_NAME ": PASS — %ux%u BGRA8 SNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA8_SNORM
   printf(PROBE_NAME ": PASS — %ux%u RGBA8 SNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA8_SINT
   printf(PROBE_NAME ": PASS — %ux%u RGBA8 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA8_UINT
   printf(PROBE_NAME ": PASS — %ux%u RGBA8 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG8_SINT
   printf(PROBE_NAME ": PASS — %ux%u RG8 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG8_UINT
   printf(PROBE_NAME ": PASS — %ux%u RG8 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R32_SINT
   printf(PROBE_NAME ": PASS — %ux%u R32 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG32_SINT
   printf(PROBE_NAME ": PASS — %ux%u RG32 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG16_UINT
   printf(PROBE_NAME ": PASS — %ux%u RG16 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG16_SINT
   printf(PROBE_NAME ": PASS — %ux%u RG16 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R32_UINT
   printf(PROBE_NAME ": PASS — %ux%u R32 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG32_UINT
   printf(PROBE_NAME ": PASS — %ux%u RG32 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R32_SFLOAT
   printf(PROBE_NAME ": PASS — %ux%u R32 SFLOAT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG32_SFLOAT
   printf(PROBE_NAME ": PASS — %ux%u RG32 SFLOAT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG16_SNORM
   printf(PROBE_NAME ": PASS — %ux%u RG16 SNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG16_UNORM
   printf(PROBE_NAME ": PASS — %ux%u RG16 UNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA16_UNORM
   printf(PROBE_NAME ": PASS — %ux%u RGBA16 UNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_SECONDARY_COMMAND_BUFFER_MULTI
   printf(PROBE_NAME ": PASS — four consecutive one-per-call "
                     "vkCmdExecuteCommands replays painted all four "
                     "quadrants exactly, 128 guard bytes unchanged, "
                     "checksum %016" PRIx64 "\n", checksum);
#elif G17P_SECONDARY_COMMAND_BUFFER
   printf(PROBE_NAME ": PASS — secondary render-pass draw produced exact "
                     "RGBA8 immediately/3s, 128 guard bytes unchanged, "
                     "checksum %016" PRIx64 "\n", checksum);
#else
   printf(PROBE_NAME ": PASS — %ux%u RGBA8, checksum "
          "%016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#endif
#endif
   return EXIT_SUCCESS;
}
