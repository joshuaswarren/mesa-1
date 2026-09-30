/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * DIAGNOSTIC EXPERIMENT ONLY -- inert unless HK_G17P_APPLE_CS or
 * HK_G17P_APPLE_FS names a file.
 *
 * Runs Apple's metal-nt applegpu_g17p machine code for the two throughput
 * benchmark kernels (sustained-max throughput.comp / throughput.frag, matched
 * by SPIR-V BLAKE3) through the ordinary Honeykrisp G17P launch path, so the
 * lab can A/B Apple's instructions against Mesa's on the same kernel, driver
 * and boot. Everything except the shader instructions and their uniform
 * inputs stays Honeykrisp's.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "vulkan/vulkan_core.h"

struct agx3_ioto_uniform_range;
struct hk_cmd_buffer;
struct hk_device;
struct hk_shader;
struct nir_shader;

/* Replace a freshly compiled variant with Apple's code when its SPIR-V
 * BLAKE3 matches and the matching environment variable names a file of the
 * expected layout. Returns VK_SUCCESS without changes for every other
 * shader. Must run before the binary is uploaded. */
VkResult hk_g17p_apple_override_apply(struct hk_device *dev,
                                      const struct nir_shader *nir,
                                      struct hk_shader *shader);

/* Apple's uniform DMA list for the compute kernel:
 *   halves  0..3   buffer(0) address (the bound SSBO's base address)
 *   halves  8..11  Params {seed, iterations} (our push constants 0..7)
 *   halves 12..15  static constants {0x61c88647, 0}
 * Returns false (and sets no ranges) if the bound state cannot supply them. */
bool hk_g17p_apple_cs_ranges(struct hk_cmd_buffer *cmd,
                             const struct hk_shader *s, uint64_t root_ptr,
                             struct agx3_ioto_uniform_range *ranges,
                             size_t capacity, size_t *range_count);

/* Apple's uniform DMA list for the fragment kernel:
 *   halves 4..7    Params {seed, iterations} (our push constants 0..7)
 *   halves 8..15   static constants {1/255, 63/255, 0x61c88647, 0} */
bool hk_g17p_apple_fs_ranges(struct hk_cmd_buffer *cmd,
                             const struct hk_shader *s, uint64_t root_ptr,
                             struct agx3_ioto_uniform_range *ranges,
                             size_t capacity, size_t *range_count);
