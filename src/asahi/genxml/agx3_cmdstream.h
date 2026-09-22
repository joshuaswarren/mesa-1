/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact, bounded AGX3 CDM command-stream helpers.
 *
 * These helpers cover only records whose byte layout is established by the
 * pinned Apple encoder corpus.  They do not enable device admission, queue
 * submission, execution, or any API capability.
 */

#pragma once

#include "agx_pack.h"
#include "../lib/agx_chip.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define AGX3_CDM_LAUNCH_SIZE 40u
#define AGX3_CDM_INDIRECT_LAUNCH_SIZE 24u
#define AGX3_CDM_INDIRECT_LAUNCH_36_SIZE 36u
#define AGX3_CDM_STREAM_LINK_SIZE 8u
#define AGX3_CDM_LINK_PADDING_SIZE 4u
#define AGX3_CDM_BARRIER_SIZE 4u
#define AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE 4u
#define AGX3_CDM_SECONDARY_BARRIER_SIZE 4u
#define AGX3_CDM_STREAM_TERMINATE_SIZE 4u

struct agx3_cdm_launch_info {
   enum agx_chip chip;
   uint32_t word0_payload;
   uint32_t word1_payload;
   uint64_t pipeline;
   uint32_t opaque_word3;
   uint32_t global_size[3];
   uint32_t local_size[3];
};

/* Exact compact launch carrying a GPU address for indirect parameters.
 *
 * The pinned performEnqueueKernel implementations all use the same six-word
 * store sequence: G15G 0x210f2d4d4, G15X 0x2125b16c8, G16G 0x2147b875c,
 * G16X 0x2152bd768, G17P 0x216ac876c, G17G 0x215e9e760, and G17X
 * 0x217626b78.  The upper half of word 4 comes from an adjacent program-
 * variant template while its lower half carries indirect_parameters[47:32].
 * Complete semantics of the raw/template words remain intentionally unnamed. */
struct agx3_cdm_indirect_launch_info {
   enum agx_chip chip;
   uint32_t word0_payload;
   uint32_t word1_payload;
   uint64_t pipeline;
   uint32_t opaque_word3;
   uint16_t indirect_word4_payload;
   uint64_t indirect_parameters;
};

/* Exact nine-word indirect launch form.  The pinned performEnqueueKernel
 * implementations use the same store order at G15G 0x210f2d508, G15X
 * 0x2125b16fc, G16G 0x2147b8790, G16X 0x2152bd79c, G17P 0x216ac87a0,
 * G17G 0x215e9e794, and G17X 0x217626bac.  Words 0..5 have the compact
 * indirect-launch layout above; words 6..8 are three independently masked
 * 11-bit values.  Their higher-level meaning is intentionally left unnamed. */
struct agx3_cdm_indirect_launch_36_info {
   enum agx_chip chip;
   uint32_t word0_payload;
   uint32_t word1_payload;
   uint64_t pipeline;
   uint32_t opaque_word3;
   uint16_t indirect_word4_payload;
   uint64_t indirect_parameters;
   uint32_t trailing_words[3];
};

struct agx3_cdm_stream_link_info {
   enum agx_chip chip;
   uint64_t target;
   bool with_return;
};

struct agx3_cdm_link_padding_info {
   enum agx_chip chip;
   uint8_t count;
};

struct agx3_cdm_barrier_info {
   enum agx_chip chip;
   /* G17 Gen7 drives bits 12, 11, 10 and 1 from flags 0..3.  Earlier
    * inspected emitters expose only canonical flag sets 0 and 0b0101. */
   uint8_t flags;
};

struct agx3_cdm_substream_barrier_info {
   enum agx_chip chip;
   uint8_t flags;
   /* Encoded as substream + 1 in bits 21..28.  Keep 255 closed because it
    * aliases the absent-field representation after the hardware mask. */
   uint8_t substream;
};

static inline void
agx3_cmdstream_store_le32(uint8_t *dst, uint32_t value)
{
   dst[0] = (uint8_t)value;
   dst[1] = (uint8_t)(value >> 8);
   dst[2] = (uint8_t)(value >> 16);
   dst[3] = (uint8_t)(value >> 24);
}

static inline uint32_t
agx3_cmdstream_load_le32(const uint8_t *src)
{
   return (uint32_t)src[0] | ((uint32_t)src[1] << 8) |
          ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
}

/* Exact launch emitters were inspected in the base and aggregate Mac plugin
 * classes listed here.  G16P and G17A stay closed because no corresponding
 * launch emitter was present in the pinned corpus. */
static inline bool
agx3_cdm_launch_target_grounded(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      return true;
   default:
      return false;
   }
}

/* Kept separate from the 40-byte direct-launch gate so later evidence for one
 * record family cannot silently widen the other.  The base and aggregate
 * G15/G16/G17P/G17 families were all inspected for this compact form. */
static inline bool
agx3_cdm_indirect_launch_target_grounded(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      return true;
   default:
      return false;
   }
}

/* Keep the nine-word family independently gated: its presence was checked at
 * each store site cited above and must not be inferred from the six-word
 * family even though their prefixes have the same byte formula. */
static inline bool
agx3_cdm_indirect_launch_36_target_grounded(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      return true;
   default:
      return false;
   }
}

/* The pinned patchCDMControlStreamAndReset sites prove the 48-bit link form
 * in every scoped base and aggregate family.  The base paths extract exactly
 * 16 high address bits before ORing type 1: G15G 0x210f7aaa4,
 * G16G 0x2147fd6f8, G17P 0x216b0d708 and G17G 0x215ee94ec. */
static inline bool
agx3_cdm_stream_link_target_grounded(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      return true;
   default:
      return false;
   }
}

static inline bool
agx3_cdm_launch_pack(const struct agx3_cdm_launch_info *info,
                     uint8_t packed[AGX3_CDM_LAUNCH_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_CDM_LAUNCH_SIZE);
   if (!info || !agx3_cdm_launch_target_grounded(info->chip) ||
       info->word0_payload >= (UINT32_C(1) << 29) ||
       info->word1_payload >= (UINT32_C(1) << 22) ||
       info->pipeline >= (UINT64_C(1) << 48) ||
       (info->pipeline & UINT64_C(0x3f)) != 0)
      return false;

   for (unsigned axis = 0; axis < 3; ++axis) {
      if (info->local_size[axis] > 0x7ff)
         return false;
   }

   struct AGX_CDM_LAUNCH_AGX3 header = {
      .word_0_payload = info->word0_payload,
      .word_1_payload = info->word1_payload,
      .pipeline_hi = info->pipeline >> 38,
      .pipeline_lo = (uint32_t)(info->pipeline >> 6),
      .opaque_word_3 = info->opaque_word3,
   };
   uint32_t header_words[AGX_CDM_LAUNCH_AGX3_LENGTH / 4] = {0};
   AGX_CDM_LAUNCH_AGX3_pack(header_words, &header);
   memcpy(packed, header_words, sizeof(header_words));

   for (unsigned axis = 0; axis < 3; ++axis) {
      agx3_cmdstream_store_le32(packed + 16 + axis * 4,
                                info->global_size[axis]);
      agx3_cmdstream_store_le32(packed + 28 + axis * 4,
                                info->local_size[axis]);
   }

   return true;
}

static inline bool
agx3_cdm_launch_unpack(enum agx_chip chip, const void *packed, size_t available,
                       struct agx3_cdm_launch_info *info)
{
   if (!info)
      return false;

   memset(info, 0, sizeof(*info));
   if (!packed || available < AGX3_CDM_LAUNCH_SIZE ||
       !agx3_cdm_launch_target_grounded(chip))
      return false;

   const uint8_t *bytes = (const uint8_t *)packed;
   const uint32_t word0 = agx3_cmdstream_load_le32(bytes);
   const uint32_t word1 = agx3_cmdstream_load_le32(bytes + 4);
   if ((word0 >> 29) != AGX_CDM_BLOCK_TYPE_LAUNCH)
      return false;

   info->chip = chip;
   info->word0_payload = word0 & ((UINT32_C(1) << 29) - 1);
   info->word1_payload = word1 & ((UINT32_C(1) << 22) - 1);
   info->pipeline = ((uint64_t)(word1 >> 22) << 38) |
                    ((uint64_t)agx3_cmdstream_load_le32(bytes + 8) << 6);
   info->opaque_word3 = agx3_cmdstream_load_le32(bytes + 12);

   for (unsigned axis = 0; axis < 3; ++axis) {
      info->global_size[axis] = agx3_cmdstream_load_le32(bytes + 16 + axis * 4);
      info->local_size[axis] = agx3_cmdstream_load_le32(bytes + 28 + axis * 4);
      if (info->local_size[axis] > 0x7ff) {
         memset(info, 0, sizeof(*info));
         return false;
      }
   }

   return true;
}

/* Pack the exact compact 24-byte indirect-parameter form.  The address field
 * is serialized as a raw 48-bit value; call-site alignment is not yet named. */
static inline bool
agx3_cdm_indirect_launch_pack(
   const struct agx3_cdm_indirect_launch_info *info,
   uint8_t packed[AGX3_CDM_INDIRECT_LAUNCH_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_CDM_INDIRECT_LAUNCH_SIZE);
   if (!info || !agx3_cdm_indirect_launch_target_grounded(info->chip) ||
       info->word0_payload >= (UINT32_C(1) << 29) ||
       info->word1_payload >= (UINT32_C(1) << 22) ||
       info->pipeline >= (UINT64_C(1) << 48) ||
       (info->pipeline & UINT64_C(0x3f)) != 0 ||
       info->indirect_parameters >= (UINT64_C(1) << 48))
      return false;

   agx3_cmdstream_store_le32(packed, info->word0_payload);
   agx3_cmdstream_store_le32(
      packed + 4,
      info->word1_payload | (uint32_t)((info->pipeline >> 38) << 22));
   agx3_cmdstream_store_le32(packed + 8,
                             (uint32_t)(info->pipeline >> 6));
   agx3_cmdstream_store_le32(packed + 12, info->opaque_word3);
   agx3_cmdstream_store_le32(
      packed + 16, (uint32_t)(info->indirect_parameters >> 32) |
                      ((uint32_t)info->indirect_word4_payload << 16));
   agx3_cmdstream_store_le32(packed + 20,
                             (uint32_t)info->indirect_parameters);
   return true;
}

/* Decode an explicitly delimited compact record.  The exact-size requirement
 * is load-bearing: the generic stream decoder must not guess whether a type-0
 * prefix belongs to this form or to the separately witnessed 36/40-byte
 * families. */
static inline bool
agx3_cdm_indirect_launch_unpack(
   enum agx_chip chip, const void *packed, size_t size,
   struct agx3_cdm_indirect_launch_info *info)
{
   if (!info)
      return false;

   memset(info, 0, sizeof(*info));
   if (!packed || size != AGX3_CDM_INDIRECT_LAUNCH_SIZE ||
       !agx3_cdm_indirect_launch_target_grounded(chip))
      return false;

   const uint8_t *bytes = (const uint8_t *)packed;
   const uint32_t word0 = agx3_cmdstream_load_le32(bytes);
   const uint32_t word1 = agx3_cmdstream_load_le32(bytes + 4);
   const uint32_t word4 = agx3_cmdstream_load_le32(bytes + 16);
   if ((word0 >> 29) != AGX_CDM_BLOCK_TYPE_LAUNCH)
      return false;

   *info = (struct agx3_cdm_indirect_launch_info){
      .chip = chip,
      .word0_payload = word0 & ((UINT32_C(1) << 29) - 1),
      .word1_payload = word1 & ((UINT32_C(1) << 22) - 1),
      .pipeline = ((uint64_t)(word1 >> 22) << 38) |
                  ((uint64_t)agx3_cmdstream_load_le32(bytes + 8) << 6),
      .opaque_word3 = agx3_cmdstream_load_le32(bytes + 12),
      .indirect_word4_payload = word4 >> 16,
      .indirect_parameters = ((uint64_t)(word4 & 0xffff) << 32) |
                             agx3_cmdstream_load_le32(bytes + 20),
   };
   return true;
}

/* Pack only an explicitly selected nine-word record. */
static inline bool
agx3_cdm_indirect_launch_36_pack(
   const struct agx3_cdm_indirect_launch_36_info *info,
   uint8_t packed[AGX3_CDM_INDIRECT_LAUNCH_36_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_CDM_INDIRECT_LAUNCH_36_SIZE);
   if (!info || !agx3_cdm_indirect_launch_36_target_grounded(info->chip) ||
       info->word0_payload >= (UINT32_C(1) << 29) ||
       info->word1_payload >= (UINT32_C(1) << 22) ||
       info->pipeline >= (UINT64_C(1) << 48) ||
       (info->pipeline & UINT64_C(0x3f)) != 0 ||
       info->indirect_parameters >= (UINT64_C(1) << 48))
      return false;

   for (unsigned word = 0; word < 3; ++word) {
      if (info->trailing_words[word] > 0x7ff)
         return false;
   }

   agx3_cmdstream_store_le32(packed, info->word0_payload);
   agx3_cmdstream_store_le32(
      packed + 4,
      info->word1_payload | (uint32_t)((info->pipeline >> 38) << 22));
   agx3_cmdstream_store_le32(packed + 8,
                             (uint32_t)(info->pipeline >> 6));
   agx3_cmdstream_store_le32(packed + 12, info->opaque_word3);
   agx3_cmdstream_store_le32(
      packed + 16, (uint32_t)(info->indirect_parameters >> 32) |
                      ((uint32_t)info->indirect_word4_payload << 16));
   agx3_cmdstream_store_le32(packed + 20,
                             (uint32_t)info->indirect_parameters);
   for (unsigned word = 0; word < 3; ++word)
      agx3_cmdstream_store_le32(packed + 24 + word * 4,
                                info->trailing_words[word]);
   return true;
}

/* Decode only an exactly delimited nine-word record.  A type-0 prefix alone
 * does not distinguish this family from the 24- and 40-byte launch forms. */
static inline bool
agx3_cdm_indirect_launch_36_unpack(
   enum agx_chip chip, const void *packed, size_t size,
   struct agx3_cdm_indirect_launch_36_info *info)
{
   if (!info)
      return false;

   memset(info, 0, sizeof(*info));
   if (!packed || size != AGX3_CDM_INDIRECT_LAUNCH_36_SIZE ||
       !agx3_cdm_indirect_launch_36_target_grounded(chip))
      return false;

   const uint8_t *bytes = (const uint8_t *)packed;
   const uint32_t word0 = agx3_cmdstream_load_le32(bytes);
   const uint32_t word1 = agx3_cmdstream_load_le32(bytes + 4);
   const uint32_t word4 = agx3_cmdstream_load_le32(bytes + 16);
   if ((word0 >> 29) != AGX_CDM_BLOCK_TYPE_LAUNCH)
      return false;

   *info = (struct agx3_cdm_indirect_launch_36_info){
      .chip = chip,
      .word0_payload = word0 & ((UINT32_C(1) << 29) - 1),
      .word1_payload = word1 & ((UINT32_C(1) << 22) - 1),
      .pipeline = ((uint64_t)(word1 >> 22) << 38) |
                  ((uint64_t)agx3_cmdstream_load_le32(bytes + 8) << 6),
      .opaque_word3 = agx3_cmdstream_load_le32(bytes + 12),
      .indirect_word4_payload = word4 >> 16,
      .indirect_parameters = ((uint64_t)(word4 & 0xffff) << 32) |
                             agx3_cmdstream_load_le32(bytes + 20),
   };

   for (unsigned word = 0; word < 3; ++word) {
      info->trailing_words[word] =
         agx3_cmdstream_load_le32(bytes + 24 + word * 4);
      if (info->trailing_words[word] > 0x7ff) {
         memset(info, 0, sizeof(*info));
         return false;
      }
   }
   return true;
}

static inline bool
agx3_cdm_stream_link_pack(
   const struct agx3_cdm_stream_link_info *info,
   uint8_t packed[AGX3_CDM_STREAM_LINK_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_CDM_STREAM_LINK_SIZE);
   if (!info || !agx3_cdm_stream_link_target_grounded(info->chip) ||
       info->target >= (UINT64_C(1) << 48))
      return false;

   struct AGX_CDM_STREAM_LINK_AGX3 link = {
      .target_hi = info->target >> 32,
      .with_return = info->with_return,
      .target_lo = (uint32_t)info->target,
   };
   uint32_t words[AGX_CDM_STREAM_LINK_AGX3_LENGTH / 4] = {0};
   AGX_CDM_STREAM_LINK_AGX3_pack(words, &link);
   memcpy(packed, words, sizeof(words));
   return true;
}

static inline bool
agx3_cdm_stream_link_unpack(
   enum agx_chip chip, const void *packed, size_t available,
   struct agx3_cdm_stream_link_info *info)
{
   if (!info)
      return false;

   memset(info, 0, sizeof(*info));
   if (!packed || available < AGX3_CDM_STREAM_LINK_SIZE ||
       !agx3_cdm_stream_link_target_grounded(chip))
      return false;

   const uint8_t *bytes = (const uint8_t *)packed;
   const uint32_t word0 = agx3_cmdstream_load_le32(bytes);
   if ((word0 >> 29) != AGX_CDM_BLOCK_TYPE_STREAM_LINK ||
       (word0 & UINT32_C(0x0fff0000)) != 0)
      return false;

   info->chip = chip;
   info->target = ((uint64_t)(word0 & 0xffff) << 32) |
                  agx3_cmdstream_load_le32(bytes + 4);
   info->with_return = (word0 >> 28) & 1;
   return true;
}

/* The AGX3 ComputeControlFlowEmitterGen1 implementations for the base and
 * aggregate G15, G16/HAL200, G17P and G17/HAL300 plugin classes all emit the
 * same four-byte type-5 link-padding token. */
static inline bool
agx3_cdm_link_padding_pack(
   const struct agx3_cdm_link_padding_info *info,
   uint8_t packed[AGX3_CDM_LINK_PADDING_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_CDM_LINK_PADDING_SIZE);
   if (!info || !agx3_cdm_launch_target_grounded(info->chip) ||
       info->count > 7)
      return false;

   struct AGX_CDM_LINK_PADDING_AGX3 padding = {
      .count = info->count,
   };
   uint32_t word = 0;
   AGX_CDM_LINK_PADDING_AGX3_pack(&word, &padding);
   agx3_cmdstream_store_le32(packed, word);
   return true;
}

static inline bool
agx3_cdm_link_padding_unpack(
   enum agx_chip chip, const void *packed, size_t available,
   struct agx3_cdm_link_padding_info *info)
{
   if (!info)
      return false;

   memset(info, 0, sizeof(*info));
   if (!packed || available < AGX3_CDM_LINK_PADDING_SIZE ||
       !agx3_cdm_launch_target_grounded(chip))
      return false;

   const uint32_t word = agx3_cmdstream_load_le32((const uint8_t *)packed);
   if ((word & ~UINT32_C(0x7)) != UINT32_C(0xa0000000))
      return false;

   info->chip = chip;
   info->count = word & 0x7;
   return true;
}

/* Exact four-byte barrier emitters are pinned in G15G/G15X, G16G/G16X,
 * G17P, G17G and G17X.  G16P and G17A remain unwitnessed here. */
static inline bool
agx3_cdm_barrier_target_grounded(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      return true;
   default:
      return false;
   }
}

static inline bool
agx3_cdm_barrier_flags_grounded(enum agx_chip chip, uint8_t flags)
{
   if (!agx3_cdm_barrier_target_grounded(chip) || flags > 0xf)
      return false;

   switch (chip) {
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      return true;
   default:
      return flags == 0 || flags == 0x5;
   }
}

static inline uint32_t
agx3_cdm_barrier_word(uint8_t flags)
{
   return UINT32_C(0x60000160) |
          ((uint32_t)(flags & 0x1) << 12) |
          ((uint32_t)(flags & 0x2) << 10) |
          ((uint32_t)(flags & 0x4) << 8) |
          ((uint32_t)(flags & 0x8) >> 2);
}

static inline bool
agx3_cdm_barrier_pack(const struct agx3_cdm_barrier_info *info,
                      uint8_t packed[AGX3_CDM_BARRIER_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_CDM_BARRIER_SIZE);
   if (!info || !agx3_cdm_barrier_flags_grounded(info->chip, info->flags))
      return false;

   agx3_cmdstream_store_le32(packed, agx3_cdm_barrier_word(info->flags));
   return true;
}

static inline bool
agx3_cdm_barrier_unpack(enum agx_chip chip, const void *packed,
                        size_t available, struct agx3_cdm_barrier_info *info)
{
   if (!info)
      return false;

   memset(info, 0, sizeof(*info));
   if (!packed || available < AGX3_CDM_BARRIER_SIZE ||
       !agx3_cdm_barrier_target_grounded(chip))
      return false;

   const uint32_t word = agx3_cmdstream_load_le32((const uint8_t *)packed);
   const uint32_t optional_mask = UINT32_C(0x00001c02);
   if ((word & ~optional_mask) != UINT32_C(0x60000160))
      return false;

   const uint8_t flags = (uint8_t)(((word >> 12) & 0x1) |
                                   ((word >> 10) & 0x2) |
                                   ((word >> 8) & 0x4) |
                                   ((word << 2) & 0x8));
   if (!agx3_cdm_barrier_flags_grounded(chip, flags))
      return false;

   info->chip = chip;
   info->flags = flags;
   return true;
}

/* HAL200's encodeSSTTFlushInvalidate emits this standalone record after the
 * CPU changes a sparse texture translation table. It is not an ordinary CDM
 * barrier: the low control byte and SSTT selector at bit 14 are distinct. */
static inline bool
agx3_cdm_sstt_flush_invalidate_pack(
   enum agx_chip chip,
   uint8_t packed[AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE);
   if (chip != AGX_CHIP_G17P)
      return false;

   agx3_cmdstream_store_le32(packed, UINT32_C(0x60024102));
   return true;
}

static inline bool
agx3_cdm_sstt_flush_invalidate_validate(enum agx_chip chip,
                                        const void *packed,
                                        size_t available)
{
   return chip == AGX_CHIP_G17P && packed &&
          available >= AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE &&
          agx3_cmdstream_load_le32(packed) == UINT32_C(0x60024102);
}

/* endVirtualSubstream and the Gen7 first-class substream emitter encode the
 * virtual substream as an eight-bit, one-biased field at bits 21..28. */
static inline bool
agx3_cdm_substream_barrier_pack(
   const struct agx3_cdm_substream_barrier_info *info,
   uint8_t packed[AGX3_CDM_BARRIER_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_CDM_BARRIER_SIZE);
   if (!info || !agx3_cdm_barrier_flags_grounded(info->chip, info->flags) ||
       info->substream == UINT8_MAX)
      return false;

   const uint32_t substream = (uint32_t)(info->substream + 1) << 21;
   agx3_cmdstream_store_le32(
      packed, agx3_cdm_barrier_word(info->flags) | substream);
   return true;
}

static inline bool
agx3_cdm_substream_barrier_unpack(
   enum agx_chip chip, const void *packed, size_t available,
   struct agx3_cdm_substream_barrier_info *info)
{
   if (!info)
      return false;

   memset(info, 0, sizeof(*info));
   if (!packed || available < AGX3_CDM_BARRIER_SIZE ||
       !agx3_cdm_barrier_target_grounded(chip))
      return false;

   const uint32_t word = agx3_cmdstream_load_le32((const uint8_t *)packed);
   const uint8_t encoded_substream = (word >> 21) & 0xff;
   const uint32_t optional_mask = UINT32_C(0x1fe01c02);
   if (encoded_substream == 0 ||
       (word & ~optional_mask) != UINT32_C(0x60000160))
      return false;

   const uint8_t flags = (uint8_t)(((word >> 12) & 0x1) |
                                   ((word >> 10) & 0x2) |
                                   ((word >> 8) & 0x4) |
                                   ((word << 2) & 0x8));
   if (!agx3_cdm_barrier_flags_grounded(chip, flags))
      return false;

   info->chip = chip;
   info->flags = flags;
   info->substream = encoded_substream - 1;
   return true;
}

/* Gen7 encodeBarrier optionally appends this fixed second barrier word.  It
 * is not present in the inspected G15/HAL200 emitters. */
static inline bool
agx3_cdm_secondary_barrier_target_grounded(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
          chip == AGX_CHIP_G17C;
}

static inline bool
agx3_cdm_secondary_barrier_pack(
   enum agx_chip chip, uint8_t packed[AGX3_CDM_SECONDARY_BARRIER_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_CDM_SECONDARY_BARRIER_SIZE);
   if (!agx3_cdm_secondary_barrier_target_grounded(chip))
      return false;

   agx3_cmdstream_store_le32(packed, UINT32_C(0x60000502));
   return true;
}

static inline bool
agx3_cdm_secondary_barrier_validate(enum agx_chip chip, const void *packed,
                                    size_t available)
{
   return packed && available >= AGX3_CDM_SECONDARY_BARRIER_SIZE &&
          agx3_cdm_secondary_barrier_target_grounded(chip) &&
          agx3_cmdstream_load_le32((const uint8_t *)packed) ==
             UINT32_C(0x60000502);
}

/* Every inspected classic and CommandEncodingNext encoder advances the CDM
 * cursor by exactly four bytes and stores this single terminate word. */
static inline bool
agx3_cdm_stream_terminate_pack(
   enum agx_chip chip, uint8_t packed[AGX3_CDM_STREAM_TERMINATE_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_CDM_STREAM_TERMINATE_SIZE);
   if (!agx3_cdm_launch_target_grounded(chip))
      return false;

   agx3_cmdstream_store_le32(packed, UINT32_C(0x40000000));
   return true;
}

static inline bool
agx3_cdm_stream_terminate_validate(enum agx_chip chip, const void *packed,
                                   size_t available)
{
   if (!packed || available < AGX3_CDM_STREAM_TERMINATE_SIZE ||
       !agx3_cdm_launch_target_grounded(chip))
      return false;

   return agx3_cmdstream_load_le32((const uint8_t *)packed) ==
          UINT32_C(0x40000000);
}
