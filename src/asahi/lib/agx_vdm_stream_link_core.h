/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 VDM Stream Link representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_STREAM_LINK_CORE_LENGTH      8u
#define AGX_VDM_STREAM_LINK_CORE_HEADER      UINT32_C(0x80000000)
#define AGX_VDM_STREAM_LINK_CORE_HEADER_MASK UINT32_C(0xffffff00)

struct agx_vdm_stream_link_core {
   uint8_t bytes[AGX_VDM_STREAM_LINK_CORE_LENGTH];
   uint8_t length;
};

/*
 * Pack/unpack the exact direct (non-returning) eight-byte VDM Stream Link
 * emitted by the inspected Apple VDMEncoderGen5 and VDMEncoderGen6 paths:
 *
 *   word 0 bits  0..7: target address bits 32..39
 *   word 0 bits  8..28: zero
 *   word 0 bits 29..31: VDM block type 4 (Stream Link)
 *   word 1 bits  0..31: target address bits 0..31
 *
 * Unlike other AGX3 command-stream pointers, this target deliberately remains
 * 40-bit through HAL300. On G17P the encoded value is a render-context-relative
 * offset, while the neighboring generations use the address domain selected
 * by their own command-stream ABI. This byte packer preserves the supplied
 * value; the caller owns that generation-specific translation. The
 * returning-call form at word 0 bit 28 is not present in the pinned AGX3
 * emitters and therefore fails closed here.
 *
 * The target allowlist contains only identities directly served by the
 * inspected G15G/G15X, G16G/G16X, G17P, and G17G/G17X Apple Metal plugins.
 * G16P and G17A remain closed despite sharing neighboring command-stream
 * generations.  This helper does not emit a live command or enable command
 * allocation, submission, or device admission.
 */
bool agx_vdm_stream_link_core_supported(enum agx_chip chip);

bool agx_vdm_stream_link_core_pack(enum agx_chip chip, uint64_t target,
                                   struct agx_vdm_stream_link_core *out);

bool agx_vdm_stream_link_core_unpack(enum agx_chip chip, const void *bytes,
                                     size_t size, uint64_t *target);

#ifdef __cplusplus
}
#endif
