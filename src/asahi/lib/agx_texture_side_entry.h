/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Recovered Apple texture-heap side-entry ABI.
 *
 * Each texture-heap slot has an eight-byte side entry alongside its read and
 * write descriptors.  Normal entries contain a low 32-bit payload and a high
 * 32-bit auxiliary/companion texture-heap slot.  Direct sparse-residency
 * entries instead contain two IEEE-754 binary32 tile dimensions.
 *
 * This interface deliberately models only independently recovered forms.
 */

#ifndef AGX_TEXTURE_SIDE_ENTRY_H
#define AGX_TEXTURE_SIDE_ENTRY_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_TEXTURE_SIDE_ENTRY_SIZE 8u

enum agx_texture_side_entry_form {
   AGX_TEXTURE_SIDE_ENTRY_FORM_NORMAL = 0,
   AGX_TEXTURE_SIDE_ENTRY_FORM_DIRECT = 2,
   /* Compatibility spelling retained for callers that still reject mode 2. */
   AGX_TEXTURE_SIDE_ENTRY_FORM_DIRECT_UNRESOLVED =
      AGX_TEXTURE_SIDE_ENTRY_FORM_DIRECT,
};

enum agx_texture_side_entry_kind {
   AGX_TEXTURE_SIDE_ENTRY_KIND_UNKNOWN = 0,
   AGX_TEXTURE_SIDE_ENTRY_KIND_BUFFER = 1,
   AGX_TEXTURE_SIDE_ENTRY_KIND_ORDINARY_IMAGE = 2,
   AGX_TEXTURE_SIDE_ENTRY_KIND_GEN9_ATOMIC_IMAGE = 3,
   AGX_TEXTURE_SIDE_ENTRY_KIND_TIER1_SPARSE_RESIDENCY = 4,
};

/* Numeric values match the recovered descriptor-generation axis. */
enum agx_texture_side_entry_generation {
   AGX_TEXTURE_SIDE_ENTRY_GENERATION_9 = 9,
   AGX_TEXTURE_SIDE_ENTRY_GENERATION_10 = 10,
};

struct agx_texture_side_entry_decoded {
   uint32_t payload;
   uint32_t auxiliary_heap_index;
};

/*
 * These class values are state[8:16]/state[8:17] values used by Apple's
 * Gen9 routine after the caller has established the shader-atomic eligibility
 * conditions.  Names reflect controlled dimension/layout correlation; they
 * do not claim a public texture-format capability.
 */
enum agx_texture_side_entry_gen9_atomic_class {
   AGX_TEXTURE_SIDE_ENTRY_GEN9_ATOMIC_CLASS_1D = 0x00100,
   AGX_TEXTURE_SIDE_ENTRY_GEN9_ATOMIC_CLASS_2D = 0x00200,
   AGX_TEXTURE_SIDE_ENTRY_GEN9_ATOMIC_CLASS_3D = 0x00300,
   AGX_TEXTURE_SIDE_ENTRY_GEN9_ATOMIC_CLASS_1D_ARRAY = 0x10100,
   AGX_TEXTURE_SIDE_ENTRY_GEN9_ATOMIC_CLASS_2D_ARRAY = 0x10200,
};

struct agx_texture_side_entry_gen9_atomic_info {
   /*
    * The caller must set this only after reproducing Apple's recovered call
    * conditions: shader-atomic usage, no metadata-bypass condition, a nonzero
    * layout-state low byte, and one of the identified classes above.
    */
   bool eligible;
   enum agx_texture_side_entry_gen9_atomic_class layout_class;

   uint64_t width_texels;
   uint64_t height_texels;
   uint64_t mip_size_bytes;
   uint64_t level_offset_bytes;
   uint64_t base_alignment_bytes;
   uint32_t format_element_bytes;
};

/*
 * Mode 2 is produced only for Apple's tier-1 sparse-residency texture.  The
 * producer stores two positive integral binary32 values at texture +0x288:
 * ordinary images use (1, 1), while texture buffers use the number of sparse
 * tiles needed along each axis.  A normal parent entry names this separate
 * mode-2 heap slot through its high auxiliary index; the mode-2 entry's own
 * high half is the height value, not another heap index.  Restrict the
 * recovered interface to values exactly representable as binary32 integers.
 */
struct agx_texture_side_entry_sparse_residency_info {
   uint64_t width_tiles;
   uint64_t height_tiles;
};

struct agx_texture_side_entry_info {
   uint32_t generation;
   enum agx_texture_side_entry_form form;
   enum agx_texture_side_entry_kind kind;
   uint32_t auxiliary_heap_index;

   union {
      uint64_t buffer_width_texels;
      struct agx_texture_side_entry_gen9_atomic_info gen9_atomic;
      struct agx_texture_side_entry_sparse_residency_info sparse_residency;
   } value;
};

bool
agx_texture_side_entry_decode(uint64_t packed,
                              struct agx_texture_side_entry_decoded *decoded);

/* Pack one of the grounded semantic forms.  On failure, packed is cleared. */
bool agx_texture_side_entry_pack(const struct agx_texture_side_entry_info *info,
                                 uint64_t *packed);

#ifdef __cplusplus
}
#endif

#endif /* AGX_TEXTURE_SIDE_ENTRY_H */
