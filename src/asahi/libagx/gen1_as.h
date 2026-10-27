/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef LIBAGX_GEN1_AS_H
#define LIBAGX_GEN1_AS_H

#ifndef __OPENCL_VERSION__
#include <stdbool.h>
#endif
#include <stdint.h>
#include "gen1_as_metadata.h"

/* agx3_envelope.comp publishes these offsets relative to the native root.
 *
 * G17P traversal takes a structure's BVH base from its AS table entry with
 * the low 14 address bits ignored, starts at the base's first 64-byte slot
 * and follows child references that are forward 64-byte slot indices from
 * that base. Nothing else in the image is read by traversal. The image itself
 * therefore has no alignment beyond its slots: it lives at the Vulkan address,
 * and a structure that is not 16 KiB-aligned gets a driver-owned base whose
 * first slot forwards to its root (hk_gen1_as_window_*). The child references
 * of every node then carry the slot distance from that base to the image.
 */
#define LIBAGX_GEN1_AS_BASE_ALIGNMENT 16384u
#define LIBAGX_GEN1_AS_SLOT_BYTES 64u
#define LIBAGX_GEN1_AS_EMPTY_BYTES 192u
#define LIBAGX_GEN1_AS_SIZE_WORD 9u
#define LIBAGX_GEN1_AS_TAIL_WORD 13u
#define LIBAGX_GEN1_AS_POINTER_WORD 16u
#define LIBAGX_GEN1_AS_BASE_WORD 18u
#define LIBAGX_GEN1_AS_ID_WORD 22u

/* Node slots (internal, triangle-leaf, AABB-leaf and instance-leaf) start
 * with one of these bytes; QTB, geometry, AABB and instance-record slots
 * start with 0x06, 0x48/0x50, 0x18 and 0x1a. Instance records' second slot
 * starts with a float and is excluded separately. */
static inline bool
libagx_gen1_as_is_node(uint32_t word)
{
   const uint32_t marker = word & 0xffu;
   return (marker >= 0xa0u && marker <= 0xa8u && !(marker & 1u)) ||
          marker == 0xe1u;
}

/* A node's first child slot is split across the low halves of its sixth and
 * seventh 64-bit words (32-bit words 10 and 12). */
static inline uint32_t
libagx_gen1_as_child_slot(uint32_t word10, uint32_t word12)
{
   return ((word10 & 0xffffu) << 16) | (word12 & 0xffffu);
}

#define LIBAGX_GEN1_AS_SERIAL_HEADER_BYTES 56u
#define LIBAGX_GEN1_AS_SERIAL_FORMAT_UUID_BYTES \
   {0x0e,0x8f,0x4b,0x53,0x91,0x1a,0x4c,0x50, \
    0x8a,0xb7,0x61,0x70,0x73,0x00,0x01,0x01}

struct libagx_gen1_as_address {
   uint64_t address;
   uint32_t resource_id;
   uint32_t reserved;
};

static inline uint64_t
libagx_gen1_as_serial_image_offset(uint64_t handles)
{
   return LIBAGX_GEN1_AS_SERIAL_HEADER_BYTES +
          LIBAGX_GEN1_AS_METADATA_STRIDE + 16u * handles;
}

static inline uint64_t
libagx_gen1_as_serial_size(const struct libagx_gen1_as_metadata *metadata,
                          uint64_t handles)
{
   return libagx_gen1_as_serial_image_offset(handles) +
          libagx_gen1_as_compacted_span(metadata);
}

static inline uint64_t
libagx_gen1_as_image_bytes(uint64_t header_size)
{
   /* Empty-AS construction writes a zero size with NaN bounds. */
   return header_size ? header_size : LIBAGX_GEN1_AS_EMPTY_BYTES;
}

static inline uint64_t
libagx_gen1_as_relocate_word(uint64_t value, uint64_t word,
                           uint64_t header_size, uint64_t destination,
                           uint32_t destination_id)
{
   /* Empty structures also retain their identity when used as instances. */
   (void)header_size;
   if (word == LIBAGX_GEN1_AS_BASE_WORD)
      return destination;
   if (word == LIBAGX_GEN1_AS_ID_WORD)
      return (value & UINT64_C(0xffffffff00000000)) | destination_id;
   return value;
}

#endif
