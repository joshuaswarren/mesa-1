/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact bounded Gen9/Gen10 PBE descriptor codec.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_PBE_AGX3_DESCRIPTOR_LENGTH 32u
#define AGX_PBE_AGX3_GENERATION_9      9u
#define AGX_PBE_AGX3_GENERATION_10     10u

/* This is intentionally the subset whose complete byte geometry is recovered:
 * ordinary non-sparse 2D images in Gen9 linear or Gen9/10 twiddled/GPU
 * memory, Gen9 GPU-tiled compressed 2D images, and Gen9's exact uncompressed
 * tiled 2D-array extension.
 * Gen9's ordinary single-layer packet has no layer-mode bit regardless of
 * allocation padding; the inherited page-aligned-layer input is ignored on
 * scalar pack and canonicalized to false on unpack. Gen10 keeps its existing
 * flag semantics. Fields with unresolved Gen9/10 semantics are not
 * represented and are required to be zero by the decoder.
 */
struct agx_pbe_agx3_2d_fields {
   uint8_t layout;
   uint8_t channels;
   uint8_t type;
   uint8_t swizzle_r;
   uint8_t swizzle_g;
   uint8_t swizzle_b;
   uint8_t swizzle_a;
   uint16_t width;
   uint16_t height;
   uint64_t buffer;
   /* Gen9 GPU-tiled 2D lossless compression metadata base. Exactly zero when
    * compressed is false; 16-byte aligned and below bit 48 otherwise. */
   uint64_t acceleration_buffer;
   bool compressed;
   /* Gen9 linear row pitch in bytes. Zero for tiled layouts. */
   uint32_t stride_B;
   /* Legacy layout metadata: encoded only for Gen10, not ordinary Gen9 2D. */
   bool page_aligned_layers;
   /* Gen9 only: convert stores to sRGB, word1 bit 61. */
   bool srgb;
};

struct agx_pbe_agx3_descriptor {
   uint8_t bytes[AGX_PBE_AGX3_DESCRIPTOR_LENGTH];
   uint8_t length;
};

bool agx_pbe_agx3_2d_pack(
   uint32_t generation,
   const struct agx_pbe_agx3_2d_fields *fields,
   struct agx_pbe_agx3_descriptor *out);

/* Gen9 HAL200 2D-array extension. PBEStateEncoderGen9 keeps the 2D field
 * geometry, selects dimension3, and places layers-1 at word1[59:44] with
 * page-aligned-layers at bit60. A one-layer array keeps dimension3. */
bool agx_pbe_agx3_2d_array_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t layers, struct agx_pbe_agx3_descriptor *out);

/* The layered packets with the native mip extension of
 * agx_pbe_agx3_2d_mip_pack(). Layers or depth, width, height and buffer
 * describe the whole chain; level selects the addressed level. */
bool agx_pbe_agx3_2d_array_mip_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t layers, uint32_t level, uint32_t levels,
   struct agx_pbe_agx3_descriptor *out);

bool agx_pbe_agx3_3d_mip_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t depth, uint32_t level, uint32_t levels,
   struct agx_pbe_agx3_descriptor *out);

/* Multisample colour target. PBEStateEncoderGen9<kAGXRevHAL200>::StateFields
 * derives the dimension nibble as `2 + ((flags >> 16) & 2)`, so an ordinary
 * non-array multisample write target selects dimension 4, and deposits the
 * two-bit sample selector as `(samples >> 2) & 3` at word0[57:56] -- the same
 * Sample Count enum the texture descriptor uses (2x=0, 4x=1, 8x=2). Every
 * other field keeps its single-sample placement: the encoder writes the
 * logical width and height, not sample-expanded ones.
 *
 * Bounded to the tiled 2x/4x colour shapes G17P's tilebuffer admits. 8x has
 * the same encoder path but no tilebuffer or layout qualification, so it is
 * refused here rather than guessed.
 */
bool agx_pbe_agx3_2d_multisample_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t samples, struct agx_pbe_agx3_descriptor *out);

/* Layered multisample colour target. The Gen9 encoder combines the layered
 * dimension-8 selector and layer-count fields with the same 2x/4x sample
 * selector as the dimension-4 packet. */
bool agx_pbe_agx3_2d_array_multisample_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t layers, uint32_t samples,
   struct agx_pbe_agx3_descriptor *out);

/* Gen9 native mip extension: the existing descriptor is unchanged for
 * level=0, levels=1. Width/height and buffer describe the complete mip chain. */
bool agx_pbe_agx3_2d_mip_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t level, uint32_t levels, struct agx_pbe_agx3_descriptor *out);

bool agx_pbe_agx3_2d_unpack(
   uint32_t generation, const void *bytes, size_t size,
   struct agx_pbe_agx3_2d_fields *fields);

#ifdef __cplusplus
}
#endif
