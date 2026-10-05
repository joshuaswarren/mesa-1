/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "util/macros.h"
#include "util/u_atomic.h"

#include "agx_pack.h"
#include "hk_native_descriptor.h"

struct hk_native_texture_descriptor {
   uint32_t opaque[HK_TEXTURE_DESCRIPTOR_AGX3_SIZE / 4];
};

struct hk_native_pbe_descriptor {
   uint32_t opaque[HK_PBE_DESCRIPTOR_AGX3_SIZE / 4];
};

bool hk_pack_native_texture(unsigned generation,
                            struct hk_native_texture_descriptor *out,
                            const struct AGX_TEXTURE *legacy);

/* Translate a multisample count only when the native sampled-texture format is
 * recovered for the descriptor generation.  There is no one-sample field in
 * the descriptor, so samples == 1 deliberately returns false.
 */
bool hk_translate_native_texture_sample_count(
   unsigned generation, unsigned samples, enum agx_sample_count *out);

/* Pack the native hardware PBE word. Gen9 supports uncompressed 2D mip chains,
 * layered 2D/3D targets, and tiled 2x/4x single- or multi-layer targets;
 * Gen10 retains its single-level subset. Software-addressed axes remain
 * separate.
 */
bool hk_pack_native_pbe(unsigned generation,
                        struct hk_native_pbe_descriptor *out,
                        const struct AGX_PBE *legacy);

/* The generation's robust null texture descriptor.  It is a pure function of
 * the generation and every sampled-image descriptor write may start from it,
 * so it is packed once per generation (hk_set_null_native_texture_slow())
 * and copied inline afterwards. */
#define HK_NULL_NATIVE_TEXTURE_GENERATIONS 16
extern struct hk_native_texture_descriptor
   hk_null_native_texture_packed[HK_NULL_NATIVE_TEXTURE_GENERATIONS];
/* 0: not packed yet, 1: being published, 2: packed valid. */
extern uint32_t hk_null_native_texture_state[HK_NULL_NATIVE_TEXTURE_GENERATIONS];

bool hk_set_null_native_texture_slow(unsigned generation,
                                     struct hk_native_texture_descriptor *out);

static inline bool
hk_set_null_native_texture(unsigned generation,
                           struct hk_native_texture_descriptor *out)
{
   if (likely(out && generation < HK_NULL_NATIVE_TEXTURE_GENERATIONS &&
              p_atomic_read(&hk_null_native_texture_state[generation]) == 2)) {
      *out = hk_null_native_texture_packed[generation];
      return true;
   }
   return hk_set_null_native_texture_slow(generation, out);
}

/* Construct the generation-native robust null PBE descriptor. */
bool hk_set_null_native_pbe(unsigned generation,
                            struct hk_native_pbe_descriptor *out);
