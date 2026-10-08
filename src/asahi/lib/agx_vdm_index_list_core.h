/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 direct VDM Index List representations.
 */

#pragma once

#ifndef __OPENCL_VERSION__
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_INDEX_LIST_DIRECT_NONINDEXED_MAX_LENGTH 20u
#define AGX_VDM_INDEX_LIST_DIRECT_INDEXED_MAX_LENGTH 32u

/* Indirect draws. The VDM record is CPU-written exactly like a direct draw;
 * only the count/instances/start live in a GPU-side argument buffer the VDM
 * dereferences at execution via the Index List "Indirect buffer" pointer.
 * G15/G16/G17P are 12 B (non-indexed) / 32 B (indexed); HAL300 (M5) appends an
 * Unk2=7 word to each, making them 16 B / 36 B. AGX2 is not a target here. */
#define AGX_VDM_INDEX_LIST_INDIRECT_NONINDEXED_MAX_LENGTH 16u
#define AGX_VDM_INDEX_LIST_INDIRECT_INDEXED_MAX_LENGTH 36u

enum agx_vdm_index_list_index_size {
   AGX_VDM_INDEX_LIST_INDEX_SIZE_U8 = 0,
   AGX_VDM_INDEX_LIST_INDEX_SIZE_U16 = 1,
   AGX_VDM_INDEX_LIST_INDEX_SIZE_U32 = 2,
};

struct agx_vdm_index_list_direct_nonindexed_info {
   uint8_t primitive;
   uint32_t count;
   uint32_t instances;
   uint32_t start;
};

struct agx_vdm_index_list_direct_indexed_info {
   uint8_t primitive;
   bool restart_enable;
   enum agx_vdm_index_list_index_size index_size;
   uint64_t index_buffer;
   uint32_t count;
   uint32_t instances;
   uint32_t start;

   /*
    * Preserved exactly as emitted after Apple's range computation.  Although
    * this word is dword-scaled, its complete byte-range semantics are not
    * independently grounded for every API edge case, so this helper does not
    * invent a byte-size conversion.
    */
   uint32_t index_buffer_size_word;
};

struct agx_vdm_index_list_direct_nonindexed {
   uint8_t bytes[AGX_VDM_INDEX_LIST_DIRECT_NONINDEXED_MAX_LENGTH];
   uint8_t length;
};

struct agx_vdm_index_list_direct_indexed {
   uint8_t bytes[AGX_VDM_INDEX_LIST_DIRECT_INDEXED_MAX_LENGTH];
   uint8_t length;
};

struct agx_vdm_index_list_indirect_nonindexed_info {
   uint8_t primitive;
   /* Native absolute argument-buffer VA. */
   uint64_t indirect_buffer;
};

struct agx_vdm_index_list_indirect_indexed_info {
   uint8_t primitive;
   bool restart_enable;
   enum agx_vdm_index_list_index_size index_size;
   uint64_t index_buffer;    /* index-buffer VA */
   /* Native absolute argument-buffer VA. */
   uint64_t indirect_buffer;
   uint32_t restart_index;   /* State Update restart value */
   uint32_t index_buffer_size_word; /* index-buffer size in dwords, minus 1 */
};

struct agx_vdm_index_list_indirect_nonindexed {
   uint8_t bytes[AGX_VDM_INDEX_LIST_INDIRECT_NONINDEXED_MAX_LENGTH];
   uint8_t length;
};

struct agx_vdm_index_list_indirect_indexed {
   uint8_t bytes[AGX_VDM_INDEX_LIST_INDIRECT_INDEXED_MAX_LENGTH];
   uint8_t length;
};

/*
 * Native command recording and the GPU-side DGC kernels must construct the
 * same words.  Keep the endian-neutral word representation inline so both
 * address spaces share the exact target gate and field layout; the host
 * wrappers below only serialize/deserialize those words.
 */
struct agx_vdm_index_list_direct_words {
   uint32_t words[AGX_VDM_INDEX_LIST_DIRECT_INDEXED_MAX_LENGTH / 4];
   uint8_t count;
};

/* Indirect records reach 9 words (36 B) on HAL300 indexed draws, one more than
 * the direct maximum, so they need their own wider carrier. */
struct agx_vdm_index_list_indirect_words {
   uint32_t words[AGX_VDM_INDEX_LIST_INDIRECT_INDEXED_MAX_LENGTH / 4];
   uint8_t count;
};

static inline bool
agx_vdm_index_list_direct_target_supported_inline(enum agx_chip chip)
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
agx_vdm_index_list_direct_is_hal300_inline(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
          chip == AGX_CHIP_G17C;
}

static inline bool
agx_vdm_index_list_index_size_supported_inline(enum agx_chip chip,
                                                uint8_t index_size)
{
   /* U8 is grounded by the native G17P/HAL200 golden.  The HAL300 captures
    * cover U16/U32 only, so keep that distinct encoding family fail-closed. */
   return index_size == AGX_VDM_INDEX_LIST_INDEX_SIZE_U16 ||
          index_size == AGX_VDM_INDEX_LIST_INDEX_SIZE_U32 ||
          (index_size == AGX_VDM_INDEX_LIST_INDEX_SIZE_U8 &&
           !agx_vdm_index_list_direct_is_hal300_inline(chip));
}

static inline bool
agx_vdm_index_list_direct_primitive_supported_inline(uint8_t primitive)
{
   switch (primitive) {
   case 0:
   case 1:
   case 3:
   case 5:
   case 6:
   case 9:
   case 11:
   case 13:
   case 14:
   case 15:
      return true;
   default:
      return false;
   }
}

static inline bool
agx_vdm_index_list_direct_pack_words(
   enum agx_chip chip, uint8_t primitive, bool indexed, bool restart_enable,
   uint8_t index_size, uint64_t index_buffer, uint32_t count,
   uint32_t instances, uint32_t start, uint32_t index_buffer_size_word,
   struct agx_vdm_index_list_direct_words *out)
{
   if (!out)
      return false;

   for (unsigned i = 0; i < AGX_VDM_INDEX_LIST_DIRECT_INDEXED_MAX_LENGTH / 4;
        ++i)
      out->words[i] = 0;
   out->count = 0;

   if (!agx_vdm_index_list_direct_target_supported_inline(chip) ||
       !agx_vdm_index_list_direct_primitive_supported_inline(primitive) ||
       instances == 0 ||
       (indexed && (!agx_vdm_index_list_index_size_supported_inline(
                       chip, index_size) ||
                    index_buffer >= ((uint64_t)1 << 48))))
      return false;

   const bool hal300 = agx_vdm_index_list_direct_is_hal300_inline(chip);
   if (!indexed) {
      out->words[0] =
         (hal300 ? 0x69c40000u : 0x61c40000u) | ((uint32_t)primitive << 8);
      out->words[1] = count;
      out->words[2] = instances;
      out->words[3] = start;
      if (hal300)
         out->words[4] = 7;
      out->count = hal300 ? 5 : 4;
      return true;
   }

   out->words[0] = (hal300 ? 0x69f00000u : 0x61f00000u) |
                   (uint32_t)((index_buffer >> 32) & (uint64_t)0xff) |
                   ((uint32_t)primitive << 8) | ((uint32_t)index_size << 17) |
                   (restart_enable ? 1u << 16 : 0);
   out->words[1] = (uint32_t)index_buffer;
   out->words[2] = count;
   out->words[3] = instances;
   out->words[4] = start;
   unsigned cursor = 5;
   if (hal300)
      out->words[cursor++] = 7;
   out->words[cursor++] = index_buffer_size_word;
   out->words[cursor++] = (uint32_t)((index_buffer >> 40) & (uint64_t)0xff);
   out->count = (uint8_t)cursor;
   return true;
}

/*
 * Indirect Index List words, mirroring agx_vdm_index_list_direct_pack_words.
 * `indirect_buffer` (A) is the argument-buffer VA; for the indexed form,
 * `index_buffer` (IB) is the index-buffer VA, `restart_index` seeds the
 * prepended 8-byte State Update, and `index_buffer_size_word` is the
 * index-buffer size in dwords minus one. The indexed record is laid out with
 * the State Update first, so word 0 here is 0x40000001.
 *
 * The indexed header is byte-identical to Apple's observed
 *   (table[prim] + (apple_index_size << 17)) & 0xffeeff00 | IB[39:32]
 *   | (restart ? bit16) | bit20
 * with table = 0x6422/0x6c22; that reduces to the additive field build below
 * because the encoded index-size field equals this enum's value (U16=1, U32=2),
 * exactly as the already-validated direct indexed packer does it.
 */
static inline bool
agx_vdm_index_list_indirect_pack_words(
   enum agx_chip chip, uint8_t primitive, bool indexed, bool restart_enable,
   uint8_t index_size, uint64_t index_buffer, uint64_t indirect_buffer,
   uint32_t restart_index, uint32_t index_buffer_size_word,
   struct agx_vdm_index_list_indirect_words *out)
{
   if (!out)
      return false;

   for (unsigned i = 0; i < AGX_VDM_INDEX_LIST_INDIRECT_INDEXED_MAX_LENGTH / 4;
        ++i)
      out->words[i] = 0;
   out->count = 0;

   if (!agx_vdm_index_list_direct_target_supported_inline(chip) ||
       !agx_vdm_index_list_direct_primitive_supported_inline(primitive) ||
       indirect_buffer >= ((uint64_t)1 << 48) || (indirect_buffer & 3u) != 0 ||
       (indexed && (!agx_vdm_index_list_index_size_supported_inline(
                       chip, index_size) ||
                    index_buffer >= ((uint64_t)1 << 48))))
      return false;

   const bool hal300 = agx_vdm_index_list_direct_is_hal300_inline(chip);
   const uint32_t addr_hi = (uint32_t)((indirect_buffer >> 32) & 0xffffu);
   const uint32_t addr_lo = (uint32_t)indirect_buffer & 0xfffffffcu;

   if (!indexed) {
      out->words[0] =
         (hal300 ? 0x6c040000u : 0x64040000u) | ((uint32_t)primitive << 8);
      out->words[1] = addr_hi;
      out->words[2] = addr_lo;
      if (hal300)
         out->words[3] = 7u;
      out->count = hal300 ? 4u : 3u;
      return true;
   }

   out->words[0] = 0x40000001u;
   out->words[1] = restart_index;
   out->words[2] = (uint32_t)(3u << 29) | (hal300 ? (1u << 27) : 0u) |
                   (1u << 26) | (1u << 21) | (1u << 20) |
                   ((uint32_t)(index_size & 7u) << 17) |
                   (restart_enable ? (1u << 16) : 0u) |
                   ((uint32_t)primitive << 8) |
                   (uint32_t)((index_buffer >> 32) & 0xffu);
   out->words[3] = (uint32_t)index_buffer;
   out->words[4] = addr_hi;
   out->words[5] = addr_lo;
   unsigned cursor = 6;
   if (hal300)
      out->words[cursor++] = 7u;
   out->words[cursor++] = index_buffer_size_word;
   out->words[cursor++] = (uint32_t)((index_buffer >> 40) & 0xffu);
   out->count = (uint8_t)cursor;
   return true;
}

bool agx_vdm_index_list_direct_nonindexed_pack(
   enum agx_chip chip,
   const struct agx_vdm_index_list_direct_nonindexed_info *info,
   struct agx_vdm_index_list_direct_nonindexed *out);

bool agx_vdm_index_list_direct_nonindexed_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_index_list_direct_nonindexed_info *info);

bool agx_vdm_index_list_direct_indexed_pack(
   enum agx_chip chip,
   const struct agx_vdm_index_list_direct_indexed_info *info,
   struct agx_vdm_index_list_direct_indexed *out);

bool agx_vdm_index_list_direct_indexed_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_index_list_direct_indexed_info *info);

bool agx_vdm_index_list_indirect_nonindexed_pack(
   enum agx_chip chip,
   const struct agx_vdm_index_list_indirect_nonindexed_info *info,
   struct agx_vdm_index_list_indirect_nonindexed *out);

bool agx_vdm_index_list_indirect_nonindexed_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_index_list_indirect_nonindexed_info *info);

bool agx_vdm_index_list_indirect_indexed_pack(
   enum agx_chip chip,
   const struct agx_vdm_index_list_indirect_indexed_info *info,
   struct agx_vdm_index_list_indirect_indexed *out);

bool agx_vdm_index_list_indirect_indexed_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_index_list_indirect_indexed_info *info);

#ifdef __cplusplus
}
#endif
