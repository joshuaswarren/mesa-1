/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include "drm-uapi/asahi_drm.h"
#include "agx_abi.h"
#include "util/bitscan.h"
#include "util/format/u_formats.h"
#include "agx_chip.h"
#include "agx_pack.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum render targets per framebuffer. This is NOT architectural, but it
 * is the ~universal API limit so there's no point in allowing more.
 */
#define AGX_MAX_RENDER_TARGETS (8)

/* Forward declarations to keep the header lean */
struct nir_shader;
struct nir_def;
struct nir_builder;

struct agx_tilebuffer_layout {
   /* Logical format of each render target. Use agx_tilebuffer_physical_format
    * to get the physical format.
    */
   enum pipe_format logical_format[AGX_MAX_RENDER_TARGETS];

   /* Which render targets are spilled. */
   bool spilled[AGX_MAX_RENDER_TARGETS];

   /* Offset into the sample of each render target. If a render target is
    * spilled, its offset is UNDEFINED. Use agx_tilebuffer_offset_B to access.
    */
   uint8_t _offset_B[AGX_MAX_RENDER_TARGETS];

   /* Total bytes per sample, rounded up as needed. Spilled render targets do
    * not count against this.
    */
   uint8_t sample_size_B;

   /* Number of samples per pixel */
   uint8_t nr_samples;

   /* If layered rendering is used */
   bool layered;

   /* Most bytes a sample may hold before render targets spill, or zero for
    * the 64-byte AGX2 bound. The tile size bounds it separately. */
   uint8_t max_sample_size_B;

   /* Selected tile size in pixels. Either 32*32, 32*16, or 16*16. */
   uint16_t tile_size;

   /* USC word corresponding to this configuration of the tilebuffer */
   struct agx_usc_shared_packed usc;
};

static inline void
agx_tilebuffer_set_drm_cmd(struct drm_asahi_cmd_render *c,
                           const struct agx_tilebuffer_layout *layout)
{
   c->utile_width_px = layout->tile_size >= 32 * 16 ? 32 : 16;
   c->utile_height_px = layout->tile_size <= 32 * 16 ? 16 : 32;
}

/*
 * _offset_B is undefined for non-spilled render targets. This safe accessor
 * asserts that render targets are not spilled rather than returning garbage.
 */
static inline uint8_t
agx_tilebuffer_offset_B(struct agx_tilebuffer_layout *layout, unsigned rt)
{
   assert(rt < AGX_MAX_RENDER_TARGETS);
   assert(!layout->spilled[rt] && "precondition");

   return layout->_offset_B[rt];
}

static inline bool
agx_tilebuffer_spills(struct agx_tilebuffer_layout *layout)
{
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (layout->spilled[rt])
         return true;
   }

   return false;
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout(const enum pipe_format *formats, uint8_t nr_cbufs,
                            uint8_t nr_samples, bool layered);

/* As agx_build_tilebuffer_layout(), spilling render targets only once a
 * sample would exceed max_sample_size_B bytes (zero for the AGX2 bound). */
struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_bounded(const enum pipe_format *formats,
                                    uint8_t nr_cbufs, uint8_t nr_samples,
                                    bool layered, uint8_t max_sample_size_B);

/* Compact attachment slots for internal programs without moving tile data.
 * map[dense slot] identifies the original logical attachment. */
unsigned agx_tilebuffer_compact(const struct agx_tilebuffer_layout *src,
                                struct agx_tilebuffer_layout *dst,
                                uint8_t map[AGX_MAX_RENDER_TARGETS]);

void agx_tilebuffer_set_samples(struct agx_tilebuffer_layout *tib,
                                unsigned nr_samples);

bool agx_nir_lower_tilebuffer(struct nir_shader *shader,
                              struct agx_tilebuffer_layout *tib,
                              uint8_t *colormasks, unsigned *bindless_base,
                              struct nir_def *write_samples, bool *translucent,
                              bool preserve_zero_mask,
                              bool lower_g17p_formats);

bool agx_nir_lower_to_per_sample(struct nir_shader *shader);

bool agx_nir_lower_monolithic_msaa(struct nir_shader *shader,
                                   uint8_t nr_samples);

/* Export registers of one sample's colours in a G17P fragment unrolled per
 * sample: eight halves for each location up to the last one written. */
static inline unsigned
agx_g17p_sample_colour_stride(uint8_t loc_written)
{
   return 8 * util_last_bit(loc_written);
}

/* Export base of sample `sample`'s depth (stencil two halves later) in such
 * a fragment, for every sample after the first: four halves per sample after
 * every sample's colours. */
static inline unsigned
agx_g17p_sample_zs_base(unsigned nr_samples, uint8_t loc_written,
                        unsigned sample)
{
   return AGX_ABI_FOUT_COLOUR(0) +
          nr_samples * agx_g17p_sample_colour_stride(loc_written) +
          4 * (sample - 1);
}

bool agx_nir_unroll_g17p_samples(struct nir_shader *shader,
                                 unsigned nr_samples, uint8_t loc_written);

bool agx_nir_hoist_g17p_zs_writes(struct nir_shader *shader);

bool agx_nir_lower_sample_intrinsics(struct nir_shader *shader,
                                     bool ignore_sample_mask_without_msaa);

uint32_t agx_tilebuffer_total_size(struct agx_tilebuffer_layout *tib);

enum pipe_format
agx_tilebuffer_physical_format(const struct agx_tilebuffer_layout *tib,
                               unsigned rt);

bool agx_tilebuffer_supports_mask(struct agx_tilebuffer_layout *tib,
                                  unsigned rt);

void agx_tilebuffer_pack_usc(struct agx_tilebuffer_layout *tib);

bool agx_g17p_rt0_tile_layout_supported(
   const struct agx_tilebuffer_layout *tib);

enum agx_g17p_mrt_layout {
   AGX_G17P_MRT_NONE,
   AGX_G17P_MRT_WBOIT,
   AGX_G17P_MRT_RGBA8,
   AGX_G17P_MRT_R11G11B10,
   AGX_G17P_MRT_RGBA32,
   AGX_G17P_MRT_R32_QUAD,
   AGX_G17P_MRT_MIXED,
};

/* Classify the byte-grounded G17P multiple-render-target layouts. The
 * background/EOT frontends use this common result to select their matching
 * shader, clear-uniform, and PBE-table contracts. */
enum agx_g17p_mrt_layout
agx_g17p_classify_mrt(const struct agx_tilebuffer_layout *tib);

#ifdef __cplusplus
} /* extern C */
#endif
