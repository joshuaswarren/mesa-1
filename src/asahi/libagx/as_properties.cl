/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl.h"
#include "gen1_as.h"

static inline uint64_t
gen1_serial_handles(global uint64_t *image,
                    struct libagx_gen1_as_metadata live)
{
   if (live.geometry_type != 2u || image[LIBAGX_GEN1_AS_SIZE_WORD] == 0)
      return 0;
   return *(global uint32_t *)((global uint8_t *)image + image[14]);
}

/* kind 0: compacted storage; kind 1: current storage span. The image needs
 * no alignment prefix: it is valid at any slot-aligned destination.
 * Live GPU metadata includes retained topology, not CPU-last-recorded flags.
 */
KERNEL(1)
libagx_gen1_as_property(global uint64_t *image, global uint64_t *metadata,
                       global uint64_t *report, uint32_t kind)
{
   global struct libagx_gen1_as_metadata *entry =
      (global struct libagx_gen1_as_metadata *)metadata;
   struct libagx_gen1_as_metadata live = *entry;
   uint64_t handles = gen1_serial_handles(image, live);
   if (kind == 2)
      *report = libagx_gen1_as_serial_size(&live, handles);
   else if (kind == 3)
      *report = handles;
   else {
      uint64_t span = kind == 0 ? libagx_gen1_as_compacted_span(&live)
                                : live.storage_bytes;
      *report = span;
   }
}

static inline uint32_t
gen1_serial_id(global struct libagx_gen1_as_address *map, uint32_t count,
               uint64_t address, volatile global uint64_t *addresses,
               uint32_t id_bound)
{
   uint32_t low = 0, high = count;
   while (low < high) {
      uint32_t mid = low + (high - low) / 2;
      if (map[mid].address < address) low = mid + 1;
      else high = mid;
   }
   if (low < count && map[low].address == address &&
       addresses[map[low].resource_id] == address)
      return map[low].resource_id;
   for (uint32_t id = 1; id < id_bound; ++id)
      if (addresses[id] == address) return id;
   return 0;
}

KERNEL(32)
libagx_gen1_as_serialize(global uint64_t *serialized, global uint64_t *image,
                         global uint64_t *metadata, global uint64_t *uuid,
                         uint64_t source_bytes)
{
   struct libagx_gen1_as_metadata live =
      *(global struct libagx_gen1_as_metadata *)metadata;
   if (live.version != LIBAGX_GEN1_AS_METADATA_VERSION ||
       live.image_bytes < LIBAGX_GEN1_AS_EMPTY_BYTES ||
       live.image_bytes > source_bytes || live.storage_bytes > source_bytes)
      return;
   uint64_t handles = gen1_serial_handles(image, live);
   if (handles > live.leaf_bound) return;
   uint64_t image_offset = libagx_gen1_as_serial_image_offset(handles);
   global uint32_t *output = (global uint32_t *)((global uint8_t *)serialized + image_offset);
   global uint32_t *input = (global uint32_t *)image;
   uint64_t first = (uint64_t)cl_global_id.x * 4;
   for (uint32_t i = 0; i < 4; ++i) {
      uint64_t word = first + i;
      if (word * 4 < live.image_bytes) output[word] = input[word];
   }
   uint64_t retained = libagx_gen1_as_compacted_retained_offset(live.image_bytes);
   global uint32_t *src_ir = (global uint32_t *)((global uint8_t *)image + live.retained_offset);
   global uint32_t *dst_ir = (global uint32_t *)((global uint8_t *)output + retained);
   for (uint32_t i = 0; i < 4; ++i) {
      uint64_t word = first + i;
      if (word * 4 < live.retained_bytes) dst_ir[word] = src_ir[word];
   }
   if (cl_global_id.x != 0) return;
   for (uint32_t i = 0; i < 4; ++i) serialized[i] = uuid[i];
   serialized[4] = libagx_gen1_as_serial_size(&live, handles);
   serialized[5] = libagx_gen1_as_compacted_span(&live);
   serialized[6] = handles;
   global uint64_t *records = serialized + 7 + handles + LIBAGX_GEN1_AS_METADATA_STRIDE / 8;
   if (handles) {
      global uint64_t *pointers = (global uint64_t *)((global uint8_t *)image + image[16]);
      uint64_t next = 0;
      for (uint32_t i = 0; i < live.leaf_bound; ++i) {
         if (pointers[2*i + 1] != 1) continue;
         uint64_t offset = pointers[2*i] - 4;
         global uint32_t *record = (global uint32_t *)((global uint8_t *)image + offset);
         serialized[7 + next] = *(global uint64_t *)((global uint8_t *)record + 88);
         records[next++] = offset;
      }
   }
   live.retained_offset = live.retained_bytes ? retained : 0;
   live.storage_bytes = libagx_gen1_as_compacted_span(&live);
   *(global struct libagx_gen1_as_metadata *)(serialized + 7 + handles) = live;
}

KERNEL(32)
libagx_gen1_as_deserialize(global uint64_t *image, global uint64_t *serialized,
                           global uint64_t *metadata, global uint64_t *uuid,
                           uint64_t destination_bytes, uint32_t destination_id)
{
   for (uint32_t i = 0; i < 4; ++i)
      if (serialized[i] != uuid[i]) return;
   uint64_t handles = serialized[6];
   if (handles > destination_bytes / 128u) return;
   struct libagx_gen1_as_metadata live =
      *(global struct libagx_gen1_as_metadata *)(serialized + 7 + handles);
   if (live.version != LIBAGX_GEN1_AS_METADATA_VERSION ||
       live.image_bytes < LIBAGX_GEN1_AS_EMPTY_BYTES ||
       live.storage_bytes > destination_bytes ||
       serialized[4] != libagx_gen1_as_serial_size(&live, handles) ||
       handles > live.leaf_bound) return;
   global uint64_t *input = (global uint64_t *)((global uint8_t *)serialized +
      libagx_gen1_as_serial_image_offset(handles));
   uint64_t first = (uint64_t)cl_global_id.x * 2;
   uint64_t header_size = input[LIBAGX_GEN1_AS_SIZE_WORD];
   for (uint32_t i = 0; i < 2; ++i) {
      uint64_t word = first + i;
      if (word * 8 < live.image_bytes)
         image[word] = libagx_gen1_as_relocate_word(input[word], word,
            header_size, (uint64_t)image, destination_id);
   }
   global uint32_t *src_ir = (global uint32_t *)((global uint8_t *)input + live.retained_offset);
   global uint32_t *dst_ir = (global uint32_t *)((global uint8_t *)image + live.retained_offset);
   first = (uint64_t)cl_global_id.x * 4;
   for (uint32_t i = 0; i < 4; ++i) {
      uint64_t word = first + i;
      if (word * 4 < live.retained_bytes) dst_ir[word] = src_ir[word];
   }
   if (cl_global_id.x == 0) {
      live.native_base = (uint64_t)image;
      live.resource_id = destination_id;
      *(global struct libagx_gen1_as_metadata *)metadata = live;
   }
}

KERNEL(32)
libagx_gen1_as_deserialize_references(global uint64_t *image,
   global uint64_t *serialized, global struct libagx_gen1_as_address *map,
   uint32_t map_count, volatile global uint64_t *addresses, uint32_t id_bound,
   global uint64_t *uuid, uint64_t destination_bytes)
{
   for (uint32_t j = 0; j < 4; ++j)
      if (serialized[j] != uuid[j]) return;
   uint64_t count = serialized[6];
   if (count > destination_bytes / 128u) return;
   struct libagx_gen1_as_metadata live =
      *(global struct libagx_gen1_as_metadata *)(serialized + 7 + count);
   if (live.version != LIBAGX_GEN1_AS_METADATA_VERSION ||
       live.storage_bytes > destination_bytes ||
       count > live.leaf_bound ||
       serialized[4] != libagx_gen1_as_serial_size(&live, count)) return;
   global uint64_t *records = serialized + 7 + count + LIBAGX_GEN1_AS_METADATA_STRIDE / 8;
   uint64_t i = cl_global_id.x;
   if (i >= count) return;
   uint64_t address = serialized[7 + i];
   global uint32_t *record = (global uint32_t *)((global uint8_t *)image + records[i]);
   record[1] = gen1_serial_id(map, map_count, address, addresses, id_bound);
   *(global uint64_t *)((global uint8_t *)record + 88) = address;
}

/* Every invocation owns disjoint image/IR pieces. CLONE keeps the retained
 * offset; COMPACT places IR immediately after the encoded image. IR is a
 * u32 array: vk_ir_header is108 bytes, so a u64 tail copy would overrun it.
 * Later consumers use the existing command completion/barrier ordering.
 */
KERNEL(32)
libagx_gen1_as_copy(global uint64_t *destination, global uint64_t *source,
                  global uint64_t *destination_metadata,
                  global uint64_t *source_metadata, uint32_t destination_id,
                  uint32_t compact)
{
   struct libagx_gen1_as_metadata live =
      *(global struct libagx_gen1_as_metadata *)source_metadata;
   uint64_t retained_offset = compact
      ? libagx_gen1_as_compacted_retained_offset(live.image_bytes)
      : live.retained_offset;
   uint64_t first = (uint64_t)cl_global_id.x * 2;
   uint64_t size = source[LIBAGX_GEN1_AS_SIZE_WORD];
   for (uint32_t i = 0; i < 2; ++i) {
      uint64_t word = first + i;
      if (word * 8 < live.image_bytes)
         destination[word] = libagx_gen1_as_relocate_word(
            source[word], word, size, (uint64_t)destination, destination_id);
   }
   global uint32_t *dst_ir = (global uint32_t *)((global uint8_t *)destination + retained_offset);
   global uint32_t *src_ir = (global uint32_t *)((global uint8_t *)source + live.retained_offset);
   first = (uint64_t)cl_global_id.x * 4;
   for (uint32_t i = 0; i < 4; ++i) {
      uint64_t word = first + i;
      if (word * 4 < live.retained_bytes)
         dst_ir[word] = src_ir[word];
   }
   if (cl_global_id.x == 0) {
      live.native_base = (uint64_t)destination;
      live.retained_offset = live.retained_bytes ? retained_offset : 0;
      live.storage_bytes = compact ? libagx_gen1_as_compacted_span(&live)
                                    : live.storage_bytes;
      live.resource_id = destination_id;
      *(global struct libagx_gen1_as_metadata *)destination_metadata = live;
   }
}

static inline global uint32_t *
gen1_rebase_image(global uint64_t *base, uint32_t serialized)
{
   return serialized
      ? (global uint32_t *)((global uint8_t *)base +
                            libagx_gen1_as_serial_image_offset(base[6]))
      : (global uint32_t *)base;
}

/* Moves an image to a structure whose traversal base is delta slots further
 * from its image (see gen1_as.h): one invocation per slot adds delta to the
 * child reference of every node slot below the tail. Serialized images are
 * stored with delta 0. Instance images also need
 * libagx_gen1_as_restore_records afterwards.
 */
KERNEL(32)
libagx_gen1_as_rebase(global uint64_t *base, uint32_t serialized, uint32_t delta)
{
   global uint32_t *image = gen1_rebase_image(base, serialized);
   global uint64_t *words = (global uint64_t *)image;
   uint64_t slot = cl_global_id.x;
   if (!words[LIBAGX_GEN1_AS_SIZE_WORD] || slot == 1 || slot == 2 ||
       slot >= words[LIBAGX_GEN1_AS_TAIL_WORD] / LIBAGX_GEN1_AS_SLOT_BYTES)
      return;
   global uint32_t *node = image + slot * (LIBAGX_GEN1_AS_SLOT_BYTES / 4);
   if (!libagx_gen1_as_is_node(node[0]))
      return;
   uint32_t child = libagx_gen1_as_child_slot(node[10], node[12]) + delta;
   node[10] = (node[10] & 0xffff0000u) | (child >> 16);
   node[12] = (node[12] & 0xffff0000u) | (child & 0xffffu);
}

/* The second slot of an instance record starts with the instance's lower x
 * bound, which libagx_gen1_as_rebase may have taken for a node marker. One
 * invocation per instance rewrites that slot from the unmodified source.
 */
KERNEL(32)
libagx_gen1_as_restore_records(global uint64_t *base, uint32_t serialized,
                               global uint64_t *source_base,
                               uint32_t source_serialized,
                               global uint64_t *metadata)
{
   struct libagx_gen1_as_metadata live =
      *(global struct libagx_gen1_as_metadata *)metadata;
   global uint8_t *image = (global uint8_t *)gen1_rebase_image(base, serialized);
   global uint8_t *source =
      (global uint8_t *)gen1_rebase_image(source_base, source_serialized);
   global uint64_t *words = (global uint64_t *)source;
   uint64_t i = cl_global_id.x;
   if (live.geometry_type != 2u || !words[LIBAGX_GEN1_AS_SIZE_WORD] ||
       i >= live.leaf_bound)
      return;
   global uint64_t *pointers =
      (global uint64_t *)(source + words[LIBAGX_GEN1_AS_POINTER_WORD]);
   if (pointers[2 * i + 1] != 1)
      return;
   uint64_t offset = pointers[2 * i] - 4 + LIBAGX_GEN1_AS_SLOT_BYTES;
   global uint64_t *dst = (global uint64_t *)(image + offset);
   global uint64_t *src = (global uint64_t *)(source + offset);
   for (uint32_t w = 0; w < LIBAGX_GEN1_AS_SLOT_BYTES / 8; ++w)
      dst[w] = src[w];
}

/* Replaces the existing empty-header GPU write; it is not another dispatch. */
KERNEL(1)
libagx_gen1_as_empty(global uint64_t *image, global uint64_t *metadata,
                    uint32_t resource_id, uint32_t build_flags, uint32_t geometry_type)
{
   global uint32_t *header = (global uint32_t *)image;
   for (uint32_t i = 0; i < LIBAGX_GEN1_AS_EMPTY_BYTES / 4; ++i)
      header[i] = i >= 20 && i < 26 ? 0x7fc00000u : 0;
   image[LIBAGX_GEN1_AS_BASE_WORD] = (uint64_t)image;
   image[LIBAGX_GEN1_AS_ID_WORD] = resource_id;
   struct libagx_gen1_as_metadata live = {
      .native_base = (uint64_t)image,
      .image_bytes = LIBAGX_GEN1_AS_EMPTY_BYTES,
      .storage_bytes = LIBAGX_GEN1_AS_EMPTY_BYTES,
      .build_flags = build_flags,
      .geometry_type = geometry_type,
      .resource_id = resource_id,
      .version = LIBAGX_GEN1_AS_METADATA_VERSION,
   };
   *(global struct libagx_gen1_as_metadata *)metadata = live;
}
