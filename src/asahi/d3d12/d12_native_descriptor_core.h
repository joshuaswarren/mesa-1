/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Generation-indexed native texture/PBE heap geometry.  This header is kept
 * independent of Mesa and Windows headers so the byte arithmetic can be
 * exhaustively host-tested.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../lib/agx_pbe_agx3_core.h"
#include "../lib/agx_texture_side_entry.h"

#define D12_NATIVE_DESCRIPTOR_GEN7  7u
#define D12_NATIVE_DESCRIPTOR_GEN8  8u
#define D12_NATIVE_DESCRIPTOR_GEN9  9u
#define D12_NATIVE_DESCRIPTOR_GEN10 10u

#define D12_NATIVE_TEXTURE_GEN9_SAMPLES_SHIFT 56u
#define D12_NATIVE_TEXTURE_GEN9_SAMPLES_MASK  UINT64_C(0x3)

/* Largest texture or PBE stride any proven layout below uses. */
#define D12_NATIVE_DESCRIPTOR_MAX_B 32u

struct d12_native_descriptor_layout {
   uint8_t texture_stride_B;
   uint8_t pbe_stride_B;
   /* Gen9/10 expose a same-index 8-byte software side entry.  This records
    * heap geometry only; D3D command submission has no proven side-heap
    * pointer channel yet and therefore does not allocate or bind one. */
   uint8_t texture_side_stride_B;
};

static inline bool
d12_native_pbe_agx3_2d_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   struct agx_pbe_agx3_descriptor *out)
{
   if (!out)
      return false;

   if (generation != D12_NATIVE_DESCRIPTOR_GEN9 &&
       generation != D12_NATIVE_DESCRIPTOR_GEN10) {
      *out = (struct agx_pbe_agx3_descriptor){0};
      return false;
   }

   return agx_pbe_agx3_2d_pack(generation, fields, out);
}

/* Layout recognition is not a device-support gate. In particular, Gen9/10
 * command streams and ISA emission remain separately fail-closed. */
static inline bool
d12_native_descriptor_layout(uint32_t generation,
                             struct d12_native_descriptor_layout *out)
{
   if (!out)
      return false;

   switch (generation) {
   case D12_NATIVE_DESCRIPTOR_GEN7:
   case D12_NATIVE_DESCRIPTOR_GEN8:
      *out = (struct d12_native_descriptor_layout){24, 24, 0};
      return true;
   case D12_NATIVE_DESCRIPTOR_GEN9:
   case D12_NATIVE_DESCRIPTOR_GEN10:
      *out = (struct d12_native_descriptor_layout){32, 32,
                                                   AGX_TEXTURE_SIDE_ENTRY_SIZE};
      return true;
   default:
      *out = (struct d12_native_descriptor_layout){0, 0, 0};
      return false;
   }
}

struct d12_native_texture_side_entry_info {
   uint32_t descriptor_generation;
   enum agx_texture_side_entry_form form;
   enum agx_texture_side_entry_kind kind;
   uint32_t auxiliary_heap_index;

   union {
      uint64_t buffer_width_texels;
      struct agx_texture_side_entry_gen9_atomic_info gen9_atomic;
      struct agx_texture_side_entry_sparse_residency_info sparse_residency;
   } value;
};

struct d12_native_texture_side_entry_decoded {
   uint32_t payload;
   uint32_t auxiliary_heap_index;
};

/* Map the bounded shared Apple side-entry ABI into the D3D descriptor core.
 * This is representation validation only: the unresolved side-heap pointer,
 * mode-2 direct value, nonzero companion construction, shader lowering and
 * command submission remain fail-closed. */
static inline bool
d12_native_texture_side_entry_pack(
   const struct d12_native_texture_side_entry_info *info, uint64_t *packed)
{
   if (!packed)
      return false;

   *packed = 0;
   if (!info)
      return false;

   uint32_t generation;
   switch (info->descriptor_generation) {
   case D12_NATIVE_DESCRIPTOR_GEN9:
      generation = AGX_TEXTURE_SIDE_ENTRY_GENERATION_9;
      break;
   case D12_NATIVE_DESCRIPTOR_GEN10:
      generation = AGX_TEXTURE_SIDE_ENTRY_GENERATION_10;
      break;
   default:
      return false;
   }

   struct agx_texture_side_entry_info shared = {
      .generation = generation,
      .form = info->form,
      .kind = info->kind,
      .auxiliary_heap_index = info->auxiliary_heap_index,
   };

   switch (info->kind) {
   case AGX_TEXTURE_SIDE_ENTRY_KIND_BUFFER:
      shared.value.buffer_width_texels = info->value.buffer_width_texels;
      break;
   case AGX_TEXTURE_SIDE_ENTRY_KIND_GEN9_ATOMIC_IMAGE:
      shared.value.gen9_atomic = info->value.gen9_atomic;
      break;
   case AGX_TEXTURE_SIDE_ENTRY_KIND_TIER1_SPARSE_RESIDENCY:
      shared.value.sparse_residency = info->value.sparse_residency;
      break;
   case AGX_TEXTURE_SIDE_ENTRY_KIND_ORDINARY_IMAGE:
   case AGX_TEXTURE_SIDE_ENTRY_KIND_UNKNOWN:
   default:
      break;
   }

   return agx_texture_side_entry_pack(&shared, packed);
}

static inline bool
d12_native_texture_side_entry_decode(
   uint64_t packed, struct d12_native_texture_side_entry_decoded *decoded)
{
   if (!decoded)
      return false;

   struct agx_texture_side_entry_decoded shared;
   if (!agx_texture_side_entry_decode(packed, &shared)) {
      *decoded = (struct d12_native_texture_side_entry_decoded){0, 0};
      return false;
   }

   decoded->payload = shared.payload;
   decoded->auxiliary_heap_index = shared.auxiliary_heap_index;
   return true;
}

/* TextureGen4::encodeImageReadState() in the independently extracted G15G,
 * G15X, G16G and G16X Apple drivers encodes ((sample_count >> 2) & 3) at
 * word 0 bits 56:57.  The field has meaning only for the two multisampled
 * texture dimensions; a non-MS texture has no one-sample selector.
 *
 * Keep this helper about native descriptor representation, not D3D12 feature
 * support.  Resource creation, rasterization and public quality-level gates
 * remain separately limited to the sample counts proven executable there.
 */
static inline bool
d12_native_texture_gen9_sample_field(bool multisampled,
                                     uint32_t sample_count,
                                     uint32_t encoded_selector,
                                     uint64_t *word0_field)
{
   if (!word0_field)
      return false;

   *word0_field = 0;
   if (!multisampled)
      return sample_count == 1 && encoded_selector == 0;

   uint64_t selector;
   switch (sample_count) {
   case 2:
      selector = 0;
      break;
   case 4:
      selector = 1;
      break;
   case 8:
      selector = 2;
      break;
   default:
      return false;
   }

   if (encoded_selector != selector)
      return false;

   *word0_field = selector << D12_NATIVE_TEXTURE_GEN9_SAMPLES_SHIFT;
   return true;
}

/* The current AGX lowering computes addresses for image atomics,
 * multisampled image stores, and typed-buffer image stores by reading fields
 * from the PBE descriptor in software.  Those helpers describe the Gen7/8
 * layout only.  A generation-sized heap offset is not sufficient to make
 * that software decode valid for Gen9/10, whose sideband layout remains
 * unproven. */
static inline bool
d12_native_pbe_software_addressing_supported(uint32_t generation)
{
   return generation == D12_NATIVE_DESCRIPTOR_GEN7 ||
          generation == D12_NATIVE_DESCRIPTOR_GEN8;
}

static inline bool
d12_native_descriptor_span(uint8_t stride_B, size_t count,
                           size_t extra_elements, size_t *size_B)
{
   if (!size_B || !stride_B || count > SIZE_MAX - extra_elements)
      return false;

   count += extra_elements;
   if (count > SIZE_MAX / stride_B)
      return false;

   *size_B = count * stride_B;
   return true;
}

static inline bool
d12_native_descriptor_offset(uint8_t stride_B, size_t index,
                             size_t *offset_B)
{
   return d12_native_descriptor_span(stride_B, index, 0, offset_B);
}

/* Gen9/10 relocated nonlinear PBE mip fields are read only when the
 * generation's mipmapped gate is asserted.  Linear descriptors do not carry
 * Level/Levels in this location. */
static inline bool
d12_native_pbe_has_levels(bool linear, uint32_t level, uint32_t levels)
{
   return !linear && (level != 0 || levels > 1);
}
