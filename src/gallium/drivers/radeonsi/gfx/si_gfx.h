/*
 * Copyright 2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef SI_GFX_H
#define SI_GFX_H

#include "util/mesa-blake3.h"
#include "util/u_stub_gfx_compute.h"

#ifdef __cplusplus
extern "C" {
#endif

struct si_screen;
struct si_shader;
struct si_shader_selector;

/* si_gfx_screen.c */
MESAPROC bool si_init_gfx_screen(struct si_screen *sscreen) TAILBT;
MESAPROC void si_fini_gfx_screen(struct si_screen *sscreen) TAILV;

/* si_shader_cache.c */
MESAPROC void si_get_ir_cache_key(struct si_shader_selector *sel, bool ngg, bool es,
                                  unsigned wave_size, unsigned char ir_blake3_cache_key[BLAKE3_KEY_LEN]) TAILV;

MESAPROC bool si_init_shader_cache(struct si_screen *sscreen) TAILB;

MESAPROC void si_init_screen_live_shader_cache(struct si_screen *sscreen) TAILV;

MESAPROC void si_destroy_shader_cache(struct si_screen *sscreen) TAILV;

MESAPROC bool si_shader_cache_load_shader(struct si_screen *sscreen, unsigned char ir_blake3_cache_key[BLAKE3_KEY_LEN],
                                          struct si_shader *shader) TAILB;

MESAPROC void si_shader_cache_insert_shader(struct si_screen *sscreen, unsigned char ir_blake3_cache_key[BLAKE3_KEY_LEN],
                                            struct si_shader *shader, bool insert_into_disk_cache) TAILV;

#ifdef __cplusplus
}
#endif

#endif /* SI_GFX_H */
