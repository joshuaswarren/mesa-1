/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact bounded packing for the constant-calculation-program suffix emitted by
 * Apple's G15/G16/G17P and HAL300 state-load encoders.
 */

#pragma once

#include "agx_chip.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_IOTO_PRESHADER_INLINE_LIMIT    256u
#define AGX_IOTO_PRESHADER_BRANCH_SIZE     10u
#define AGX_IOTO_PRESHADER_PRELUDE_SIZE    2u
#define AGX_IOTO_PRESHADER_ABSENT_SIZE     4u
#define AGX_IOTO_PRESHADER_MAX_SUFFIX_SIZE AGX_IOTO_PRESHADER_INLINE_LIMIT
#define AGX_IOTO_PRESHADER_NO_VA           UINT64_MAX
#define AGX_IOTO_PRESHADER_NO_BRANCH       SIZE_MAX

enum agx_ioto_preshader_mode {
   AGX_IOTO_PRESHADER_MODE_ABSENT = 0,
   AGX_IOTO_PRESHADER_MODE_INLINE,
   AGX_IOTO_PRESHADER_MODE_EXTERNAL,
};

struct agx_ioto_preshader_desc {
   /* Host-visible compiled Ioto bytes beginning at the program start. */
   const void *code;
   size_t code_size;

   /* GPU base plus program start, or AGX_IOTO_PRESHADER_NO_VA. */
   uint64_t external_va;

   /* ConstantCalculationProgramInfo bytes +0x28 and +0x29, respectively. */
   bool force_external;
   bool request_prelude;

   /* Encoder state at +0x100 (GPU0) or +0x130 (GPU1). */
   bool suppress_prelude;
};

struct agx_ioto_preshader_output {
   enum agx_ioto_preshader_mode mode;
   size_t length;

   /* Offset within bytes[] of B, or AGX_IOTO_PRESHADER_NO_BRANCH. */
   size_t branch_offset;
   uint8_t bytes[AGX_IOTO_PRESHADER_MAX_SUFFIX_SIZE];
};

/*
 * Reproduce the size choice in setConstantCalculationProgramInfo:
 *
 *   end32 > start ? (uint32_t)(end32 - start) : alternate_end - start
 *
 * Unlike Apple's unchecked subtraction, malformed descending ranges fail.
 * `size` is unchanged on failure.
 */
bool agx_ioto_preshader_program_size(uint32_t start, uint32_t end32,
                                     uint64_t alternate_end, size_t *size);

/*
 * Bind a nonempty compiled range to matching CPU and GPU copies of its
 * original binary. The GPU base must not refer to a separately linked main
 * shader. Both copies must contain the whole range, and its GPU extent must
 * fit the even-addressed 48-bit branch domain.
 *
 * Updates only code, code_size, and external_va. The caller's explicit
 * force/prelude options are preserved; no ABI policy is inferred here.
 * The descriptor is unchanged on failure, including a missing GPU mapping.
 * This validates bounds, not instructions, register/scratch use, or BO lifetime.
 */
bool agx_ioto_preshader_bind_range(const void *binary, size_t binary_size,
                                   uint64_t binary_va, size_t gpu_size,
                                   size_t offset, size_t size,
                                   struct agx_ioto_preshader_desc *desc);

/*
 * Pack the target-family long B or BL form.  The constant-calculation suffix
 * calls this with link=false; BL is used by the separate profile-state loader.
 * Addresses must be even and fit in 48 bits.  `out` is unchanged on failure.
 */
bool
agx_ioto_preshader_pack_branch(enum agx_chip chip, uint64_t target_va,
                               bool link,
                               uint8_t out[AGX_IOTO_PRESHADER_BRANCH_SIZE]);

/*
 * Pack only the suffix appended by finish() after resource/state loads,
 * LoadShader and any PIXREL record:
 *
 *   inline:  compiled bytes with every trailing 06 00 NOP removed
 *   external: optional fixed LDIMM followed by the long target-family B
 *   absent:   0e 00 00 00
 *
 * Inline selection is `code_size <= 256 && !force_external`.  A zero-length
 * result after NOP trimming takes the absent path.  `output` is unchanged on
 * every failure.
 */
bool agx_ioto_preshader_emit(enum agx_chip chip,
                             const struct agx_ioto_preshader_desc *desc,
                             struct agx_ioto_preshader_output *output);

#ifdef __cplusplus
}
#endif
