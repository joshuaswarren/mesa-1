/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

/* Keep image-plane bases representable by every texture/PBE path. */
#define HK_IMAGE_PLANE_ALIGN_B 128u

/* Linear texture descriptors encode bytes-per-row in 16-byte units. */
#define HK_IMAGE_LINEAR_STRIDE_ALIGN_B 16u

bool hk_image_explicit_linear_stride(uint64_t row_pitch_B,
                                     uint64_t minimum_row_pitch_B,
                                     uint32_t *stride_B_out);

bool hk_image_explicit_plane_range(uint64_t offset_B, uint64_t size_B,
                                   uint64_t *end_B_out);

bool hk_image_plane_ranges_overlap(uint64_t offset_a_B, uint64_t size_a_B,
                                   uint64_t offset_b_B, uint64_t size_b_B);

bool hk_image_implicit_plane_range(uint64_t cursor_B, uint64_t size_B,
                                   uint64_t *offset_B_out, uint64_t *end_B_out);

bool hk_image_plane_binding(uint64_t bo_address, uint64_t bo_size_B,
                            uint64_t memory_offset_B, uint64_t plane_offset_B,
                            uint64_t plane_size_B, uint64_t *address_out,
                            uint64_t *bo_offset_out);
