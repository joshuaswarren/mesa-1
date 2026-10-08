/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact, target-gated semantic transform recovered from Apple's AGX3
 * fragment imageblock constructor and encodeImageBlockData helper.
 *
 * This is a research/validation surface.  It does not emit an Ioto record,
 * alter a state tail, submit work, or admit an AGX3 device.
 */

#pragma once

#include "agx_chip.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum agx_ioto_imageblock_encoding {
   AGX_IOTO_IMAGEBLOCK_ENCODING_UNSUPPORTED = 0,
   AGX_IOTO_IMAGEBLOCK_ENCODING_GPU0,
   AGX_IOTO_IMAGEBLOCK_ENCODING_GPU1,
};

struct agx_ioto_imageblock_params {
   uint32_t p0;
   uint32_t p1;
   uint32_t p2;
   uint32_t p3;
   uint32_t p4;
};

/*
 * Exact constructor intermediates.  Every arithmetic member is the result of
 * the corresponding Apple 32-bit operation, including modulo-2^32 wrap.
 * `lo` and `hi` are zero when `active` is false; the remaining intermediates
 * are retained so boundary behavior can be validated independently.
 */
struct agx_ioto_imageblock_result {
   enum agx_chip chip;
   enum agx_ioto_imageblock_encoding encoding;

   uint32_t sample_count;
   uint32_t p0_aligned;
   uint32_t first_product;

   uint32_t a;
   uint32_t x;
   uint32_t y;
   uint32_t four_a;
   uint32_t second_product;

   uint32_t x_log_code;
   uint32_t y_log_code;
   uint32_t sample_code;

   uint32_t rounded_four_a;
   uint32_t q;
   uint32_t q_field;

   bool active;
   uint64_t lo;
   uint64_t hi;

   uint32_t h_payload;
   uint64_t h;
};

#ifdef __cplusplus
}
#endif
