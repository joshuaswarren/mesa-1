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

#ifdef __cplusplus
}
#endif
